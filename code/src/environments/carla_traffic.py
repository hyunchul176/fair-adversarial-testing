"""
CARLA Traffic Manager for Adversarial Scenario Generation.

PURPOSE: Manage NPC (background vehicle) spawning, cruise control with PID,
and IDM car-following in CARLA 0.9.16 for the ASG framework.

RESEARCH CONTEXT:
- Adapted from Frenet_revised's TrafficManager for CARLA 0.9.16
- NPCs maintain lane (constant Frenet d) and follow IDM car-following model
- The RL-controlled adversarial vehicle is separate; this module handles
  the naturalistic background traffic that provides realistic context
- Realistic NPC behavior is critical: unrealistic background traffic
  invalidates the adversarial scenarios (AuthSim, T-ITS 2025)

INTEGRATION:
- Uses Spline3D global route for Frenet<->Cartesian conversions
- PID controllers track waypoints along the spline
- IDM provides speed adaptation to preceding vehicles
- LineOfSightSensor (CARLA obstacle detector) feeds into IDM

CARLA 0.9.16 COMPATIBILITY:
- Uses set_target_velocity (0.9.16+) with fallback to set_velocity
- Uses try_spawn_actor for safe spawning without exceptions
- autopilot=False: all NPC control is manual via PID+IDM
"""

from __future__ import annotations

import carla
import math
import random
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Tuple, Any
import logging

from src.environments.cubic_spline_planner import Spline3D, frenet_to_inertial

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_speed(vehicle) -> float:
    """
    Get vehicle speed in m/s from CARLA velocity vector.

    Args:
        vehicle: CARLA vehicle actor

    Returns:
        Speed magnitude in m/s
    """
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def set_velocity_compat(actor, velocity: carla.Vector3D) -> None:
    """
    Set actor velocity with CARLA 0.9.16 compatibility.

    Tries set_target_velocity first (0.9.16+), falls back to set_velocity
    for older CARLA versions.

    Args:
        actor: CARLA actor
        velocity: Target velocity as carla.Vector3D
    """
    try:
        actor.set_target_velocity(velocity)
    except AttributeError:
        try:
            actor.set_velocity(velocity)
        except AttributeError:
            logger.warning(
                "Neither set_target_velocity nor set_velocity available "
                "for actor %s", actor.id
            )


# ---------------------------------------------------------------------------
# PID Longitudinal Controller
# ---------------------------------------------------------------------------

class PIDLongitudinalController:
    """
    Longitudinal PID controller for vehicle speed tracking.

    Uses a standard PID formulation to compute throttle commands that
    track a target speed. The default gains (K_P=40.0, K_I=4.0, K_D=0.1)
    are tuned for CARLA's throttle response characteristics, where
    throttle in [0, 1] maps to a large force range.

    RESEARCH NOTE:
    High K_P is necessary because CARLA throttle [0,1] produces large
    forces, so the speed error (m/s) needs amplification. These gains
    were empirically tuned in the Frenet_revised project.
    """

    def __init__(
        self,
        vehicle,
        K_P: float = 40.0,
        K_D: float = 0.1,
        K_I: float = 4.0,
        dt: float = 0.1,
    ):
        """
        Args:
            vehicle: CARLA vehicle actor
            K_P: Proportional gain (high because CARLA throttle [0,1] -> large force)
            K_D: Derivative gain for damping speed oscillations
            K_I: Integral gain for steady-state error elimination
            dt: Control timestep in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._error_integral: float = 0.0
        self._previous_error: float = 0.0

    def reset(self) -> None:
        """Reset accumulated PID state."""
        self._error_integral = 0.0
        self._previous_error = 0.0

    def run_step(self, target_speed: float) -> Tuple[float, float]:
        """
        Compute throttle to track target speed.

        Args:
            target_speed: Desired speed in m/s

        Returns:
            (throttle, current_speed): throttle in [0, 1], current speed in m/s
        """
        current_speed = get_speed(self._vehicle)
        throttle = self._pid_control(target_speed, current_speed)
        return throttle, current_speed

    def _pid_control(self, target_speed: float, current_speed: float) -> float:
        """
        Standard PID control law for throttle.

        Args:
            target_speed: Desired speed in m/s
            current_speed: Current speed in m/s

        Returns:
            Throttle command clipped to [0, 1]
        """
        error = target_speed - current_speed

        # Proportional term
        p_term = self._K_P * error

        # Integral term (with anti-windup via output clamping)
        self._error_integral += error * self._dt
        i_term = self._K_I * self._error_integral

        # Derivative term
        d_term = self._K_D * (error - self._previous_error) / self._dt if self._dt > 0 else 0.0
        self._previous_error = error

        throttle = p_term + i_term + d_term
        return float(np.clip(throttle, 0.0, 1.0))


# ---------------------------------------------------------------------------
# PID Lateral Controller
# ---------------------------------------------------------------------------

class PIDLateralController:
    """
    Lateral PID controller for steering to track waypoints.

    Computes steering angle to reach a target waypoint using cross-track
    error projected onto the vehicle's lateral axis. The default gains
    (K_P=0.3, K_D=0.0, K_I=0.0) provide stable lane-keeping for NPC
    vehicles that don't change lanes.

    RESEARCH NOTE:
    NPCs in our framework maintain constant Frenet d (no lane changes).
    The lateral controller only corrects drift and handles road curvature.
    """

    def __init__(
        self,
        vehicle,
        K_P: float = 0.3,
        K_D: float = 0.0,
        K_I: float = 0.0,
        dt: float = 0.1,
    ):
        """
        Args:
            vehicle: CARLA vehicle actor
            K_P: Proportional gain for cross-track error
            K_D: Derivative gain (default 0 for smooth NPC steering)
            K_I: Integral gain (default 0; NPC steering needs no integral action)
            dt: Control timestep in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._error_integral: float = 0.0
        self._previous_error: float = 0.0

    def reset(self) -> None:
        """Reset accumulated PID state."""
        self._error_integral = 0.0
        self._previous_error = 0.0

    def run_step(self, waypoint: List[float]) -> float:
        """
        Compute steering angle to reach the target waypoint.

        Uses cross-track error: the signed lateral distance between the
        vehicle's forward axis and the vector to the waypoint, combined
        with the heading error.

        Args:
            waypoint: Target position [x, y] in world coordinates

        Returns:
            Steering command clipped to [-1, 1]
        """
        transform = self._vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        # Vehicle heading in radians (CARLA yaw is in degrees)
        vehicle_yaw = math.radians(rotation.yaw)

        # Vector from vehicle to waypoint
        dx = waypoint[0] - location.x
        dy = waypoint[1] - location.y

        # Cross-track error: project (dx, dy) onto the vehicle's lateral axis
        # Lateral axis is perpendicular to heading: (-sin(yaw), cos(yaw))
        # Positive cross_track_error means waypoint is to the left
        cross_track_error = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        # Heading error: angle between vehicle heading and direction to waypoint
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - vehicle_yaw
        # Normalize to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # Combined error (cross-track dominates at distance, heading at proximity)
        distance_to_wp = math.sqrt(dx ** 2 + dy ** 2)
        if distance_to_wp > 0.5:
            # At distance, use heading error (which naturally incorporates cross-track)
            error = heading_error
        else:
            # Very close: use cross-track error directly
            error = cross_track_error

        # PID computation
        p_term = self._K_P * error

        self._error_integral += error * self._dt
        i_term = self._K_I * self._error_integral

        d_term = self._K_D * (error - self._previous_error) / self._dt if self._dt > 0 else 0.0
        self._previous_error = error

        steering = p_term + i_term + d_term
        return float(np.clip(steering, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Combined Vehicle PID Controller
# ---------------------------------------------------------------------------

class VehiclePIDController:
    """
    Combined lateral (steering) and longitudinal (throttle) PID controller.

    Wraps PIDLongitudinalController and PIDLateralController to produce
    a complete carla.VehicleControl command from a target speed and waypoint.

    RESEARCH NOTE:
    This follows the standard CARLA PID agent pattern but with gains tuned
    for highway NPC behavior (higher longitudinal gains for responsive
    speed tracking, lower lateral gains for smooth lane-keeping).
    """

    def __init__(
        self,
        vehicle,
        args_lateral: Optional[Dict[str, float]] = None,
        args_longitudinal: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            vehicle: CARLA vehicle actor
            args_lateral: Dict with keys K_P, K_D, K_I for lateral PID
                         (defaults: K_P=0.3, K_D=0.0, K_I=0.0)
            args_longitudinal: Dict with keys K_P, K_D, K_I for longitudinal PID
                              (defaults: K_P=40.0, K_D=0.1, K_I=4.0)
        """
        self._vehicle = vehicle

        if args_lateral is None:
            args_lateral = {"K_P": 0.3, "K_D": 0.0, "K_I": 0.0}
        if args_longitudinal is None:
            args_longitudinal = {"K_P": 40.0, "K_D": 0.1, "K_I": 4.0}

        self._lon_controller = PIDLongitudinalController(
            vehicle,
            K_P=args_longitudinal.get("K_P", 40.0),
            K_D=args_longitudinal.get("K_D", 0.1),
            K_I=args_longitudinal.get("K_I", 4.0),
        )
        self._lat_controller = PIDLateralController(
            vehicle,
            K_P=args_lateral.get("K_P", 0.3),
            K_D=args_lateral.get("K_D", 0.0),
            K_I=args_lateral.get("K_I", 0.0),
        )

    def reset(self) -> None:
        """Reset both PID controllers."""
        self._lon_controller.reset()
        self._lat_controller.reset()

    def run_step(
        self, target_speed: float, waypoint: List[float]
    ) -> carla.VehicleControl:
        """
        Compute a complete vehicle control command.

        Args:
            target_speed: Desired speed in m/s
            waypoint: Target waypoint [x, y] in world coordinates

        Returns:
            carla.VehicleControl with throttle, steer, and brake
        """
        throttle, current_speed = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step(waypoint)

        # Smooth throttle/brake transition — no dead zone
        speed_error = target_speed - current_speed
        control = carla.VehicleControl()
        if speed_error >= 0:
            control.throttle = throttle
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(1.0, -speed_error * 0.3)
        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False

        return control


# ---------------------------------------------------------------------------
# Intelligent Driver Model (IDM)
# ---------------------------------------------------------------------------

class IntelligentDriverModel:
    """
    Intelligent Driver Model for car-following behavior.

    Implements Treiber et al.'s IDM to compute desired speed based on
    the gap and relative velocity to the preceding vehicle. When no
    vehicle is ahead, the model accelerates toward the desired free-flow speed.

    RESEARCH NOTE:
    IDM is the standard car-following model used in both highway-env
    and CARLA NPC control. The default parameters (v0=13.89 m/s, T=1.5s)
    produce moderate, non-aggressive driving behavior. This is intentional:
    background NPCs should be naturalistic, not aggressive. The adversarial
    behavior is only from the RL-controlled vehicle.

    Reference:
        Treiber, M., Hennecke, A., & Helbing, D. (2000).
        Congested traffic states in empirical observations and microscopic
        simulations. Physical Review E, 62(2), 1805.
    """

    def __init__(
        self,
        vehicle,
        v0: float = 25.6,
        T: float = 1.71,
        a_max: float = 2.0,
        b: float = 3.0,
        s0: float = 10.5,
        delta: int = 4,
    ):
        """
        Args:
            vehicle: CARLA vehicle actor
            v0: Desired free-flow speed [m/s] (HighD mean=25.6, NDD-calibrated)
            T: Safe time headway [s]
            a_max: Maximum acceleration [m/s^2]
            b: Comfortable braking deceleration [m/s^2]
            s0: Minimum gap [m] (jam distance)
            delta: Acceleration exponent (higher = sharper transition to free-flow)
        """
        self._vehicle = vehicle
        self.v0 = v0
        self.T = T
        self.a_max = a_max
        self.b = b
        self.s0 = s0
        self.delta = delta

    def run_step(
        self,
        vd: float,
        vehicle_ahead: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute desired speed using IDM.

        Args:
            vd: Free-flow desired speed [m/s] (may differ from v0 for variety)
            vehicle_ahead: Dict with 'distance' [m] and 'speed' [m/s] of the
                          preceding vehicle, or None if no vehicle ahead

        Returns:
            Desired speed [m/s] after applying IDM acceleration for one step
        """
        current_speed = get_speed(self._vehicle)

        if vehicle_ahead is None:
            # Free-flow: accelerate toward desired speed
            accel = self.a_max * (1.0 - (current_speed / max(vd, 0.1)) ** self.delta)
        else:
            s = vehicle_ahead["distance"]
            v_lead = vehicle_ahead["speed"]

            # Ensure positive gap
            s = max(s, 0.1)

            # Approaching speed (positive when closing in)
            delta_v = current_speed - v_lead

            # Desired dynamical gap
            s_star = (
                self.s0
                + max(0.0, current_speed * self.T
                      + (current_speed * delta_v) / (2.0 * math.sqrt(self.a_max * self.b)))
            )

            # IDM acceleration
            accel = self.a_max * (
                1.0
                - (current_speed / max(vd, 0.1)) ** self.delta
                - (s_star / s) ** 2
            )

        # Clamp acceleration to physical limits
        accel = np.clip(accel, -self.b * 2.0, self.a_max)

        # Convert acceleration to desired speed (single step)
        dt = 0.1  # control timestep
        desired_speed = max(0.0, current_speed + accel * dt)

        return desired_speed


# ---------------------------------------------------------------------------
# Cruise Control (per-vehicle controller)
# ---------------------------------------------------------------------------

class CruiseControl:
    """
    Per-NPC controller combining PID + IDM for cruise control.

    Each NPC vehicle gets a CruiseControl instance that:
    1. Queries its LineOfSightSensor for preceding vehicle detection
    2. Uses IDM to compute a target speed considering car-following
    3. Uses PID to generate throttle/steering commands
    4. Tracks a waypoint along the global spline at constant Frenet d

    NPCs do NOT change lanes (constant d). Lane changes are reserved for
    the RL-controlled adversarial vehicle.

    RESEARCH NOTE:
    This cruise control mimics naturalistic highway driving. The combination
    of IDM + PID is standard in microscopic traffic simulation (SUMO, CARLA TM).
    The key difference from CARLA's built-in Traffic Manager is that we have
    full control over the car-following parameters and can extract per-vehicle
    state for aggressiveness analysis.
    """

    def __init__(
        self,
        vehicle,
        los_sensor,
        s: float,
        d: float,
        lane: int,
        world_module,
        target_speed: float,
        dt: float = 0.1,
    ):
        """
        Args:
            vehicle: CARLA vehicle actor
            los_sensor: LineOfSightSensor for detecting vehicle ahead
            s: Current Frenet longitudinal position [m]
            d: Frenet lateral offset [m] (constant for NPCs)
            lane: Lane index (0-3)
            world_module: Reference to world module (provides town_map, csp, etc.)
            target_speed: Cruise speed [m/s]
            dt: Control timestep [s]
        """
        self.vehicle = vehicle
        self.vehicleController = VehiclePIDController(vehicle)
        self.IDM = IntelligentDriverModel(vehicle)
        self.los_sensor = los_sensor
        self.s = s
        self.d = d
        self.lane = lane
        self.world_module = world_module
        self.target_speed = target_speed
        self.dt = dt

        # State tracking
        self._prev_speed: float = 0.0
        self._prev_location: Optional[carla.Location] = None

    def tick(self, ego_vehicle: Optional[Any] = None) -> List[float]:
        """
        Execute one control step for this NPC vehicle.

        Pipeline:
        1. Get current state (location, speed, yaw)
        2. Get next waypoint from the global spline (fixed d, look-ahead s)
        3. Query LOS sensor for vehicle ahead
        3b. If ego_vehicle is provided and threatens (cut-in or close lateral
            approach), override vehicle_ahead with ego — makes the NPC IDM
            brake-react like highway-env IDM. The LOS sensor is forward-only
            and misses ego behind/beside the NPC, which lets adversarial RL
            converge to fast-crash. This override restores the sustained
            interaction signal observed in highway-env.
        4. Compute IDM target speed
        5. Apply PID control
        6. Apply control to vehicle

        Args:
            ego_vehicle: Optional reference to the ego (RL-controlled)
                vehicle. When provided, the NPC's IDM treats ego as the
                leading vehicle when ego is ahead within ~25 m or laterally
                close within ~3 m (cut-in geometry).

        Returns:
            [x, y, z, speed, acceleration, yaw] - current vehicle state
        """
        # 1. Current state
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        current_speed = get_speed(self.vehicle)
        yaw = math.radians(rotation.yaw)

        # Estimate acceleration
        acceleration = (current_speed - self._prev_speed) / self.dt if self.dt > 0 else 0.0
        self._prev_speed = current_speed

        # 2. Get next waypoint from global spline
        # Look ahead proportional to speed — at highway speeds (30 m/s),
        # need at least 1.5-2s preview = 45-60m
        look_ahead = max(10.0, min(60.0, current_speed * 2.0))
        waypoint_s = self.s + look_ahead

        # Clamp to spline bounds
        csp = TrafficManager.global_csp
        if csp is not None:
            max_s = csp.s[-1]
            if waypoint_s > max_s:
                waypoint_s = max_s - 1.0

            wp_x, wp_y, wp_z, wp_yaw = frenet_to_inertial(waypoint_s, self.d, csp)
            waypoint = [wp_x, wp_y]
        else:
            # Fallback: use CARLA map waypoint
            carla_wp = self.world_module.world.get_map().get_waypoint(
                location, project_to_road=True
            )
            if carla_wp is not None:
                next_wps = carla_wp.next(look_ahead)
                if next_wps:
                    wp_loc = next_wps[0].transform.location
                    waypoint = [wp_loc.x, wp_loc.y]
                else:
                    waypoint = [
                        location.x + look_ahead * math.cos(yaw),
                        location.y + look_ahead * math.sin(yaw),
                    ]
            else:
                waypoint = [
                    location.x + look_ahead * math.cos(yaw),
                    location.y + look_ahead * math.sin(yaw),
                ]

        # 3. Query LOS sensor for vehicle ahead
        vehicle_ahead = None
        if self.los_sensor is not None:
            try:
                detection = self.los_sensor.get_detection()
                if detection is not None:
                    vehicle_ahead = {
                        "distance": detection.get("distance", 100.0),
                        "speed": detection.get("speed", self.target_speed),
                    }
            except (AttributeError, TypeError):
                # Sensor may not be ready or may have different API
                pass

        # 3b. Ego threat-aware override: if ego is ahead of this NPC or
        #     laterally cutting in, treat ego as the leading vehicle. The
        #     LOS sensor only catches forward dynamic actors; without this
        #     branch the NPC ignores adversarial ego approaching from behind
        #     or alongside, which makes the adversarial agent converge to fast crashes. This
        #     restores the highway-env IDM-style reactive behavior.
        if ego_vehicle is not None and ego_vehicle.is_alive:
            try:
                ego_tf = ego_vehicle.get_transform()
                ego_vel = ego_vehicle.get_velocity()
                ego_speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)

                # NPC heading and lateral axis (right-hand rule)
                fwd_x = math.cos(yaw)
                fwd_y = math.sin(yaw)
                right_x = math.cos(yaw - math.pi / 2)
                right_y = math.sin(yaw - math.pi / 2)

                dx = ego_tf.location.x - location.x
                dy = ego_tf.location.y - location.y
                longitudinal = dx * fwd_x + dy * fwd_y  # >0: ego ahead of NPC
                lateral = abs(dx * right_x + dy * right_y)

                # Cut-in geometry: ego is roughly in-lane (|lat| < 3 m) and
                # ahead or alongside (longitudinal > -2 m, within 25 m).
                ego_is_threat = (
                    -2.0 < longitudinal < 25.0
                    and lateral < 3.0
                )

                if ego_is_threat:
                    # Distance metric for IDM: clamp longitudinal so the NPC
                    # never sees ego as "behind" (negative distance breaks IDM)
                    ego_dist = max(longitudinal, 1.0)
                    # If LOS already has a closer NPC ahead, keep that. Otherwise
                    # inject ego as leading.
                    if vehicle_ahead is None or vehicle_ahead["distance"] > ego_dist:
                        vehicle_ahead = {
                            "distance": ego_dist,
                            "speed": ego_speed,
                        }
            except (AttributeError, RuntimeError):
                pass

        # 4. IDM target speed
        idm_speed = self.IDM.run_step(self.target_speed, vehicle_ahead)

        # 5. Apply PID control
        control = self.vehicleController.run_step(idm_speed, waypoint)

        # Junction avoidance for rightmost lane (lane 2)
        # At junctions, NPCs in the rightmost lane tend to drift off
        # the highway. Increase lateral correction to stay in lane.
        if self.lane == 2:
            carla_wp = self.world_module.world.get_map().get_waypoint(
                location, project_to_road=True
            )
            if carla_wp is not None and carla_wp.is_junction:
                # Strengthen steering toward lane center
                control.steer = float(np.clip(control.steer * 1.5, -1.0, 1.0))

        # 6. Apply control
        self.vehicle.apply_control(control)

        self._prev_location = location

        return [
            location.x,
            location.y,
            location.z,
            current_speed,
            acceleration,
            math.degrees(yaw),
        ]

    def update_s(self, s: float) -> None:
        """
        Update the Frenet longitudinal position estimate.

        Called by TrafficManager after global s estimation.

        Args:
            s: Updated Frenet s coordinate
        """
        self.s = s


# ---------------------------------------------------------------------------
# Traffic Manager
# ---------------------------------------------------------------------------

class TrafficManager:
    """
    Manages all NPC (background) vehicles in the CARLA simulation.

    Handles spawning, despawning, and per-tick control of background vehicles
    using CruiseControl (PID + IDM). Vehicles are spawned on a grid in
    Frenet coordinates and maintain their lanes throughout the episode.

    RESEARCH NOTE:
    Background traffic realism matters for adversarial scenario validity.
    If NPCs behave unrealistically, the adversarial scenarios generated
    against them lack transferability to real-world conditions. Using IDM
    with calibrated parameters ensures NPC behavior matches observed
    highway driving patterns from HighD data.

    SPAWNING STRATEGY:
    Grid-based spawning in Frenet coordinates:
    - 20 columns (longitudinal): s in [ego_s - 20, ego_s + 170], 10m spacing
    - 4 rows (lanes): d corresponding to lane indices [-1, 0, 1, 2]
    - N_SPAWN_CARS random positions selected from the 80-cell grid
    """

    # Class-level global spline reference (set before spawning)
    global_csp: Optional[Spline3D] = None

    def __init__(
        self,
        name: str,
        world_module,
        n_spawn_cars: int = 15,
        min_speed: float = 21.0,
        max_speed: float = 30.0,
        lane_width: float = 3.5,
        max_s: float = 3000.0,
        track_length: float = 750.0,
    ):
        """
        Args:
            name: Identifier for this traffic manager
            world_module: Reference to the world module (provides CARLA world,
                         map, sensors, etc.)
            n_spawn_cars: Number of NPC vehicles to spawn (default 15)
            min_speed: Minimum NPC cruise speed [m/s] (default 25.0 = 90 km/h, NDD-calibrated)
            max_speed: Maximum NPC cruise speed [m/s] (default 34.0 = 122 km/h, NDD-calibrated)
            lane_width: Lane width [m] for Frenet d computation
            max_s: Maximum Frenet s on the route [m]
            track_length: Length of the active simulation track [m]
        """
        self.name = name
        self.world_module = world_module
        self.N_SPAWN_CARS = n_spawn_cars
        self.MIN_SPEED = min_speed
        self.MAX_SPEED = max_speed
        self.LANE_WIDTH = lane_width
        self.MAX_S = max_s
        self.TRACK_LENGTH = track_length

        # Actor storage: list of dicts with keys:
        # {Actor, Sensor, CruiseControl, FrenetState}
        self.actors_batch: List[Dict[str, Any]] = []

        # CARLA references (set in start())
        self._world = None
        self._blueprint_library = None
        self._vehicle_blueprints: List = []

        # Available vehicle colors for visual variety
        self._colors = [
            "255,0,0", "0,255,0", "0,0,255", "255,255,0",
            "255,0,255", "0,255,255", "128,128,128", "255,128,0",
            "128,0,255", "0,128,255", "255,255,255", "64,64,64",
        ]

        logger.info(
            "TrafficManager '%s' initialized: n_spawn=%d, speed=[%.1f, %.1f] m/s",
            name, n_spawn_cars, min_speed, max_speed,
        )

    def start(self) -> None:
        """
        Initialize CARLA world references and filter vehicle blueprints.

        Must be called after CARLA world is connected and before reset().
        Filters blueprints to only include 4-wheeled vehicles (no bikes,
        motorcycles, etc.) for realistic highway traffic.
        """
        self._world = self.world_module.world
        self._blueprint_library = self._world.get_blueprint_library()

        # Filter to 4-wheel vehicles only
        all_vehicles = self._blueprint_library.filter("vehicle.*")
        self._vehicle_blueprints = []
        for bp in all_vehicles:
            # Exclude bicycles, motorcycles, and other 2-wheelers
            if bp.has_attribute("number_of_wheels"):
                n_wheels = int(bp.get_attribute("number_of_wheels"))
                if n_wheels >= 4:
                    self._vehicle_blueprints.append(bp)
            else:
                # If attribute missing, include it (most cars have 4 wheels)
                self._vehicle_blueprints.append(bp)

        if not self._vehicle_blueprints:
            logger.error("No suitable vehicle blueprints found!")
            # Fallback to all vehicles
            self._vehicle_blueprints = list(all_vehicles)

        logger.info(
            "TrafficManager started: %d vehicle blueprints available",
            len(self._vehicle_blueprints),
        )

    def reset(self, ego_s: float, ego_d: float) -> None:
        """
        Reset traffic: destroy existing actors and spawn new NPCs.

        Uses grid-based spawning in Frenet coordinates:
        - Grid covers s in [ego_s - 20, ego_s + 170] with 10m spacing (20 columns)
        - 4 lane rows: lane indices [-1, 0, 1, 2]
        - Randomly selects N_SPAWN_CARS positions from the 80-cell grid
        - Avoids spawning in the ego vehicle's grid cell

        Args:
            ego_s: Ego vehicle's current Frenet s position
            ego_d: Ego vehicle's current Frenet d position (used to identify
                   ego's lane for spawn avoidance)
        """
        # Destroy existing actors
        self.destroy()

        if TrafficManager.global_csp is None:
            logger.error(
                "global_csp not set! Call update_global_route_csp() before reset()."
            )
            return

        # Build spawn grid — NDD-calibrated spacing (HighD DHW 25th pctl = 23m)
        # Concentrated near ego for interaction density
        s_start = ego_s - 30.0
        s_end = ego_s + 200.0
        s_step = 23.0  # HighD DHW 25th percentile
        s_positions = np.arange(s_start, s_end + s_step, s_step)

        # 4 lane rows
        lane_indices = [-1, 0, 1, 2]

        # Build all grid cells as (s, lane) pairs, skipping high-curvature zones
        grid_cells: List[Tuple[float, int]] = []
        for s_pos in s_positions:
            s_clamped = float(np.clip(s_pos, 0.0, TrafficManager.global_csp.s[-1] - 1.0))
            # Skip curves where PID steering is unstable
            try:
                curvature = abs(TrafficManager.global_csp.calc_curvature(s_clamped))
                if curvature > 0.01:  # radius < 100m
                    continue
            except (TypeError, ValueError):
                pass
            for lane in lane_indices:
                grid_cells.append((s_clamped, lane))

        # Identify ego's approximate lane
        ego_lane = round(ego_d / self.LANE_WIDTH) if self.LANE_WIDTH > 0 else 0

        # Remove cells too close to ego (same lane only, within 20m)
        safe_cells = []
        for s_pos, lane in grid_cells:
            if lane == ego_lane and abs(s_pos - ego_s) < 20.0:
                continue
            safe_cells.append((s_pos, lane))

        # Randomly select N_SPAWN_CARS positions
        n_to_spawn = min(self.N_SPAWN_CARS, len(safe_cells))
        if n_to_spawn == 0:
            logger.warning("No safe spawn cells available!")
            return

        selected_cells = random.sample(safe_cells, n_to_spawn)

        # Spawn vehicles
        spawn_count = 0
        for s_pos, lane in selected_cells:
            target_speed = random.uniform(self.MIN_SPEED, self.MAX_SPEED)
            actor_entry = self.spawn_one_actor(s_pos, lane, target_speed)
            if actor_entry is not None:
                spawn_count += 1

        logger.info(
            "TrafficManager reset: spawned %d/%d NPCs around ego_s=%.1f",
            spawn_count, n_to_spawn, ego_s,
        )

    def spawn_one_actor(
        self, s: float, lane: int, target_speed: float
    ) -> Optional[Dict[str, Any]]:
        """
        Spawn a single NPC vehicle at the given Frenet position.

        Converts Frenet (s, d) to Cartesian, spawns the vehicle with a
        random blueprint and color, attaches a LineOfSightSensor, and
        creates a CruiseControl instance.

        Args:
            s: Frenet longitudinal position [m]
            lane: Lane index (-1, 0, 1, 2)
            target_speed: Cruise target speed [m/s]

        Returns:
            Actor entry dict or None if spawn failed
        """
        if TrafficManager.global_csp is None:
            logger.error("Cannot spawn: global_csp is None")
            return None

        # Convert lane index to Frenet d
        d = lane * self.LANE_WIDTH

        # Frenet to Cartesian
        try:
            x, y, z, yaw = frenet_to_inertial(s, d, TrafficManager.global_csp)
        except (ValueError, TypeError) as e:
            logger.warning("Frenet conversion failed at s=%.1f, d=%.1f: %s", s, d, e)
            return None

        if x is None or y is None:
            logger.warning("Frenet conversion returned None at s=%.1f", s)
            return None

        # Create spawn transform
        spawn_transform = carla.Transform(
            carla.Location(x=x, y=y, z=z + 0.1),  # +0.1m minimal offset
            carla.Rotation(yaw=math.degrees(yaw)),
        )

        # Select random blueprint and color
        bp = random.choice(self._vehicle_blueprints)
        if bp.has_attribute("color"):
            color = random.choice(self._colors)
            bp.set_attribute("color", color)

        # Try to spawn the vehicle (safe: returns None on failure)
        actor = self._world.try_spawn_actor(bp, spawn_transform)
        if actor is None:
            logger.debug(
                "Spawn failed at s=%.1f, lane=%d (likely collision with existing actor)",
                s, lane,
            )
            return None

        # Disable autopilot (we control manually via PID+IDM).
        # Wrap in try-except: newly spawned actors are not registered with
        # the TM, so set_autopilot(False) may trigger an RPC timeout on the
        # 'unregister_vehicle' call — safe to ignore.
        try:
            actor.set_autopilot(False)
        except RuntimeError as e:
            logger.debug("set_autopilot(False) RPC timeout (safe to ignore): %s", e)

        # Set initial velocity — start at moderate speed and let IDM
        # accelerate to target. Starting at full highway speed causes
        # instability before PID stabilizes steering.
        init_v = min(target_speed, 20.0)  # Cap initial at 20 m/s (72 km/h)
        vx = init_v * math.cos(yaw)
        vy = init_v * math.sin(yaw)
        set_velocity_compat(actor, carla.Vector3D(x=vx, y=vy, z=0.0))

        # Attach LineOfSightSensor for vehicle-ahead detection
        los_sensor = self._attach_los_sensor(actor)

        # Create CruiseControl instance
        cruise = CruiseControl(
            vehicle=actor,
            los_sensor=los_sensor,
            s=s,
            d=d,
            lane=lane,
            world_module=self.world_module,
            target_speed=target_speed,
        )

        # Build actor entry
        actor_entry = {
            "Actor": actor,
            "Sensor": los_sensor,
            "CruiseControl": cruise,
            "FrenetState": {"s": s, "d": d, "lane": lane},
        }
        self.actors_batch.append(actor_entry)

        logger.debug(
            "Spawned NPC id=%d at s=%.1f, lane=%d, speed=%.1f m/s",
            actor.id, s, lane, target_speed,
        )
        return actor_entry

    def _attach_los_sensor(self, vehicle) -> Optional[Any]:
        """
        Attach a LineOfSightSensor (obstacle detector) to a vehicle.

        Uses CARLA's built-in obstacle detection sensor to detect the
        vehicle directly ahead. This feeds into IDM for car-following.

        Args:
            vehicle: CARLA vehicle actor to attach sensor to

        Returns:
            Sensor actor or None if attachment failed
        """
        try:
            # Use CARLA's obstacle detection sensor
            bp = self._blueprint_library.find("sensor.other.obstacle")
            bp.set_attribute("distance", "50.0")  # Detection range [m]
            bp.set_attribute("hit_radius", "1.0")  # Detection width [m]
            bp.set_attribute("only_dynamics", "true")  # Only detect moving objects
            bp.set_attribute("sensor_tick", "0.1")  # Update rate [s]

            # Mount at front of vehicle, looking forward
            sensor_transform = carla.Transform(
                carla.Location(x=2.5, y=0.0, z=0.5)
            )
            sensor = self._world.try_spawn_actor(
                bp, sensor_transform, attach_to=vehicle
            )

            if sensor is not None:
                # Store detection results — initialize to None (no vehicle ahead)
                # so IDM starts in free-flow mode, not phantom car-following
                sensor._detection_data = None
                sensor._detection_tick = 0

                def _on_obstacle(event, _sensor=sensor):
                    if event.other_actor is not None:
                        _sensor._detection_data = {
                            "distance": max(event.distance, 0.1),
                            "speed": get_speed(event.other_actor),
                        }
                        _sensor._detection_tick = 0

                sensor.listen(_on_obstacle)

                # Expire stale detections (>5 ticks = 0.5s without update)
                def _get_detection(_sensor=sensor):
                    _sensor._detection_tick += 1
                    if _sensor._detection_tick > 5:
                        _sensor._detection_data = None  # Expired
                    return _sensor._detection_data

                sensor.get_detection = _get_detection

            return sensor

        except (RuntimeError, AttributeError) as e:
            logger.warning("Failed to attach LOS sensor: %s", e)
            return None

    def set_ego_vehicle(self, ego_vehicle: Optional[Any]) -> None:
        """Register the ego (RL-controlled) vehicle so NPCs' IDM can react.

        Without an ego reference, NPCs rely only on the forward-only
        LineOfSightSensor and miss adversarial ego approaching from behind
        or alongside, which makes the adversarial agent converge to fast crashes. With this
        registration, each NPC's CruiseControl.tick receives the ego handle
        and injects ego as the leading vehicle when cut-in geometry holds.
        """
        self._ego_vehicle = ego_vehicle

    def tick(self) -> List[List[float]]:
        """
        Execute one control step for all NPC vehicles.

        For each NPC:
        1. Run CruiseControl.tick(ego_vehicle=...) to compute and apply
           control. Ego handle is forwarded so NPCs' IDM treats ego as the
           leading vehicle on cut-in/close approach.
        2. Update Frenet s estimate from current Cartesian position

        Returns:
            List of [x, y, z, speed, acceleration, yaw] for each NPC
        """
        states = []
        csp = TrafficManager.global_csp
        ego = getattr(self, "_ego_vehicle", None)

        for entry in self.actors_batch:
            actor = entry["Actor"]
            cruise = entry["CruiseControl"]
            frenet_state = entry["FrenetState"]

            # Check if actor is still alive
            if not actor.is_alive:
                continue

            # Run cruise control with ego threat-awareness
            try:
                state = cruise.tick(ego_vehicle=ego)
                states.append(state)
            except RuntimeError as e:
                logger.warning("CruiseControl tick failed for actor %d: %s", actor.id, e)
                continue

            # Update Frenet s estimate
            if csp is not None:
                location = actor.get_transform().location
                yaw_deg = actor.get_transform().rotation.yaw
                new_s = self.estimate_s(
                    frenet_state["s"],
                    location.x,
                    location.y,
                    math.radians(yaw_deg),
                )
                frenet_state["s"] = new_s
                cruise.update_s(new_s)

        return states

    def destroy(self) -> None:
        """
        Destroy all NPC actors and their attached sensors.

        Safely handles already-destroyed actors and cleans up
        the actors_batch list.
        """
        destroy_count = 0
        for entry in self.actors_batch:
            # Destroy sensor first
            sensor = entry.get("Sensor")
            if sensor is not None:
                try:
                    if sensor.is_alive:
                        sensor.stop()
                        sensor.destroy()
                except RuntimeError:
                    pass

            # Destroy vehicle actor
            actor = entry.get("Actor")
            if actor is not None:
                try:
                    if actor.is_alive:
                        actor.destroy()
                        destroy_count += 1
                except RuntimeError:
                    pass

        self.actors_batch.clear()
        logger.info("TrafficManager destroyed %d NPC actors", destroy_count)

    def estimate_s(
        self, s_prev: float, x: float, y: float, yaw: float
    ) -> float:
        """
        Estimate current Frenet s from Cartesian position.

        Uses a normal vector projection method: finds the closest point
        on the spline to (x, y) near the previous s estimate, then
        projects the displacement onto the tangent direction to refine.

        This is more efficient than a full spline search because NPC
        positions change smoothly between ticks.

        Args:
            s_prev: Previous Frenet s estimate [m]
            x: Current x position in world coordinates
            y: Current y position in world coordinates
            yaw: Current heading in radians

        Returns:
            Updated Frenet s estimate [m]
        """
        csp = TrafficManager.global_csp
        if csp is None:
            return s_prev

        max_s = csp.s[-1]

        # Search in a local window around previous s
        search_range = 20.0  # NPCs move at most ~15m between ticks at 150km/h
        ds = 0.5  # Search resolution

        s_min = max(0.0, s_prev - search_range * 0.3)  # Asymmetric: NPCs mostly move forward
        s_max = min(max_s, s_prev + search_range)

        best_s = s_prev
        best_dist_sq = float("inf")

        s_test = s_min
        while s_test <= s_max:
            pos = csp.calc_position(s_test)
            if pos[0] is None:
                s_test += ds
                continue

            ix, iy, _ = pos
            dist_sq = (x - ix) ** 2 + (y - iy) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_s = s_test
            s_test += ds

        # Refine using tangent projection at best_s
        pos = csp.calc_position(best_s)
        if pos[0] is not None:
            ix, iy, _ = pos
            ref_yaw = csp.calc_yaw(best_s)

            # Tangent vector at best_s
            tx = math.cos(ref_yaw)
            ty = math.sin(ref_yaw)

            # Project displacement onto tangent
            dx = x - ix
            dy = y - iy
            projection = dx * tx + dy * ty

            best_s += projection

        # Clamp to valid range
        best_s = float(np.clip(best_s, 0.0, max_s))

        return best_s

    def get_actor_states(self) -> List[Dict[str, Any]]:
        """
        Get current state of all managed NPC vehicles.

        Returns:
            List of dicts with keys: id, x, y, z, speed, yaw, s, d, lane
        """
        states = []
        for entry in self.actors_batch:
            actor = entry["Actor"]
            if not actor.is_alive:
                continue

            transform = actor.get_transform()
            location = transform.location
            speed = get_speed(actor)
            frenet = entry["FrenetState"]

            states.append({
                "id": actor.id,
                "x": location.x,
                "y": location.y,
                "z": location.z,
                "speed": speed,
                "yaw": transform.rotation.yaw,
                "s": frenet["s"],
                "d": frenet["d"],
                "lane": frenet["lane"],
            })

        return states

    @classmethod
    def update_global_route_csp(cls, csp: Spline3D) -> None:
        """
        Set the global cubic spline reference path for all traffic managers.

        Must be called before reset() / spawn operations. The spline
        defines the reference path (road centerline) used for Frenet
        coordinate conversions.

        Args:
            csp: Spline3D representing the global route
        """
        cls.global_csp = csp
        logger.info(
            "Global route CSP updated: s_max=%.1f m",
            csp.s[-1] if csp is not None else 0.0,
        )
