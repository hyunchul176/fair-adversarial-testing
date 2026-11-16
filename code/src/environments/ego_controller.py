"""
High-level PID Waypoint Controller for CARLA Ego Vehicle.

PURPOSE:
    Converts high-level RL actions (target_speed, lane_offset) into low-level
    CARLA VehicleControl commands (throttle, steer, brake) via PID waypoint
    tracking in the Frenet frame.

RESEARCH CONTEXT:
    The RL agent operates in a goal-conditioned action space where it decides
    WHAT to do (desired speed, desired lateral position).  This controller
    handles the HOW -- tracking the desired trajectory using longitudinal and
    lateral PID loops.  Decoupling intent from execution is standard practice
    in adversarial scenario generation (AuthSim, T-ITS 2025; HAD-Gen, IV 2024)
    and prevents the agent from exploiting low-level physics glitches that
    would make generated scenarios unrealistic.

ARCHITECTURE:
    Mode A (legacy target_speed):
        RL Agent --> [target_speed, lane_offset] --> EgoController.compute_control()
                                                          |
                                                          v
                                            [throttle, steer, brake]

    Mode B (acceleration action, current):
        RL Agent --> [acceleration]   --> direct throttle/brake (env handles)
                 --> [lane_offset]    --> EgoController.compute_steering()
                                                          |
                                                          v
                                                       [steer]

    The controller uses the global cubic spline (Spline3D from
    cubic_spline_planner.py) to convert Frenet (s, d) targets into Cartesian
    waypoints, then steers toward them while maintaining the commanded speed
    (Mode A) or only the commanded lateral offset (Mode B).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

try:
    import carla
except ImportError:
    carla = None  # Allow importing for type-checking without CARLA installed

from src.environments.cubic_spline_planner import Spline3D, frenet_to_inertial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default PID gains -- intentionally conservative to avoid oscillation.
# These should be tuned per vehicle model / simulation timestep.
# ---------------------------------------------------------------------------
_DEFAULT_LON_KP: float = 2.0
_DEFAULT_LON_KI: float = 0.5
_DEFAULT_LON_KD: float = 0.1
_DEFAULT_LAT_KP: float = 1.0
_DEFAULT_LAT_KI: float = 0.0  # No integral for lateral -- avoids windup drift
_DEFAULT_LAT_KD: float = 0.3

# Safety clamps
_MIN_LOOK_AHEAD_M: float = 5.0
_LOOK_AHEAD_TIME_S: float = 1.0  # look-ahead = max(5, speed * 1.0)
_SPLINE_END_MARGIN_M: float = 1.0  # Stay away from spline boundary


class PIDController:
    """
    Simple PID controller with anti-windup clamping.

    Used for both longitudinal (speed) and lateral (heading) tracking.
    The integral term is clamped to ``integral_limit`` to prevent windup
    when the error is sustained (e.g., heavy braking from high speed).
    """

    def __init__(
        self,
        K_p: float,
        K_i: float,
        K_d: float,
        dt: float,
        integral_limit: float = 5.0,
    ) -> None:
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.dt = dt
        self.integral_limit = integral_limit

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._initialized: bool = False

    def reset(self) -> None:
        """Clear accumulated state.  Call at the start of each episode."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def step(self, error: float) -> float:
        """
        Compute the PID output for the current error.

        Args:
            error: Signed error (desired - actual).

        Returns:
            Control command (unbounded; caller should clamp as needed).
        """
        # Proportional
        p_term = self.K_p * error

        # Integral with anti-windup clamp
        self._integral += error * self.dt
        self._integral = max(
            -self.integral_limit, min(self.integral_limit, self._integral)
        )
        i_term = self.K_i * self._integral

        # Derivative (skip on first call to avoid a spike)
        if self._initialized and self.dt > 0.0:
            derivative = (error - self._prev_error) / self.dt
        else:
            derivative = 0.0
            self._initialized = True
        d_term = self.K_d * derivative

        self._prev_error = error

        return p_term + i_term + d_term


class EgoController:
    """
    Converts high-level RL commands (target speed + lateral offset) into
    low-level CARLA VehicleControl via PID waypoint tracking.

    The RL agent decides **what** to do (speed, lane position).
    This controller decides **how** to do it (throttle, steering, braking).

    Coordinates are expressed in the Frenet frame defined by
    ``global_csp`` (a ``Spline3D`` representing the global highway route).

    Args:
        vehicle: CARLA vehicle actor to control.
        global_csp: Pre-computed cubic spline for the reference path.
        lane_width: Lane width in meters (default 3.5, HighD standard).
        dt: Simulation timestep in seconds (must match CARLA
            ``fixed_delta_seconds``).
        lon_gains: Tuple of (K_p, K_i, K_d) for the speed PID.
        lat_gains: Tuple of (K_p, K_i, K_d) for the heading PID.
    """

    def __init__(
        self,
        vehicle,                     # carla.Vehicle (typed loosely for importability)
        global_csp: Spline3D,
        lane_width: float = 3.5,
        dt: float = 0.1,
        lon_gains: Optional[Tuple[float, float, float]] = None,
        lat_gains: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if vehicle is None:
            raise ValueError("vehicle must not be None")
        if global_csp is None:
            raise ValueError("global_csp (Spline3D) must not be None")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.vehicle = vehicle
        self.global_csp = global_csp
        self.lane_width = lane_width
        self.dt = dt

        # Longitudinal PID (speed tracking)
        lp, li, ld = lon_gains or (_DEFAULT_LON_KP, _DEFAULT_LON_KI, _DEFAULT_LON_KD)
        self.lon_pid = PIDController(K_p=lp, K_i=li, K_d=ld, dt=dt)

        # Lateral PID (waypoint heading tracking)
        sp, si, sd = lat_gains or (_DEFAULT_LAT_KP, _DEFAULT_LAT_KI, _DEFAULT_LAT_KD)
        self.lat_pid = PIDController(K_p=sp, K_i=si, K_d=sd, dt=dt)

        # Internal Frenet state
        self._current_s: float = 0.0
        self._max_s: float = global_csp.s[-1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initial_s: float) -> None:
        """
        Reset controller state at the start of an episode.

        Args:
            initial_s: Starting Frenet s-position of the ego vehicle.
        """
        self._current_s = max(0.0, min(initial_s, self._max_s))
        self.lon_pid.reset()
        self.lat_pid.reset()

    def update_s(self, s: float) -> None:
        """
        Update the controller's estimate of the ego's Frenet s-position.

        The caller (environment) is responsible for computing s each tick
        via ``inertial_to_frenet_approx`` and passing it here.

        Args:
            s: Current longitudinal arc-length position on the reference spline.
        """
        self._current_s = max(0.0, min(s, self._max_s))

    def get_speed(self) -> float:
        """
        Return the current speed (m/s) of the controlled vehicle.

        Computed from the CARLA velocity vector (ignores vertical component
        because highway driving is planar).
        """
        velocity = self.vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2)

    def compute_control(
        self,
        target_speed: float,
        target_d: float,
    ):  # -> carla.VehicleControl (typed loosely for importability)
        """
        Convert high-level RL commands into a CARLA VehicleControl.

        Args:
            target_speed: Desired speed in m/s.  Reasonable range: [0, 40].
                          Values are clamped to [0, 50] internally.
            target_d: Desired lateral offset from the reference path center
                      in meters.  Positive = left of reference path.
                      Typical range: [-lane_width, +lane_width]
                      i.e. [-3.5, 3.5] for single-lane offsets.

        Returns:
            A ``carla.VehicleControl`` namedtuple-like object with throttle,
            steer, and brake fields set appropriately.

        Raises:
            RuntimeError: If the CARLA Python API is not available.
        """
        if carla is None:
            raise RuntimeError(
                "CARLA Python API is not installed.  Cannot create VehicleControl."
            )

        # ---- 0. Clamp inputs ----
        target_speed = max(0.0, min(target_speed, 50.0))

        # ---- 1. Current vehicle state ----
        transform = self.vehicle.get_transform()
        current_speed = self.get_speed()
        current_yaw = math.radians(transform.rotation.yaw)

        # ---- 2. Generate a target waypoint in world coordinates ----
        look_ahead = max(_MIN_LOOK_AHEAD_M, current_speed * _LOOK_AHEAD_TIME_S)
        target_s = self._current_s + look_ahead

        # Clamp to stay within spline bounds
        target_s = min(target_s, self._max_s - _SPLINE_END_MARGIN_M)
        target_s = max(0.0, target_s)

        try:
            wp_x, wp_y, _wp_z, _wp_yaw = frenet_to_inertial(
                target_s, target_d, self.global_csp
            )
        except ValueError:
            # Spline evaluation failed (typically near boundaries).
            # Fall back to a straight-ahead waypoint at the look-ahead distance.
            logger.warning(
                "frenet_to_inertial failed at s=%.1f, d=%.2f; "
                "falling back to straight-ahead waypoint.",
                target_s,
                target_d,
            )
            wp_x = transform.location.x + look_ahead * math.cos(current_yaw)
            wp_y = transform.location.y + look_ahead * math.sin(current_yaw)

        # ---- 3. Longitudinal control (speed PID) ----
        speed_error = target_speed - current_speed
        accel_cmd = self.lon_pid.step(speed_error)

        # ---- 4. Lateral control (heading-error PID) ----
        dx = wp_x - transform.location.x
        dy = wp_y - transform.location.y
        dist_to_wp = math.sqrt(dx * dx + dy * dy)

        if dist_to_wp < 1e-4:
            # Essentially on top of the waypoint; no heading correction needed.
            heading_error = 0.0
        else:
            target_yaw = math.atan2(dy, dx)
            heading_error = self._normalize_angle(target_yaw - current_yaw)

        steer_cmd = self.lat_pid.step(heading_error)

        # ---- 5. Build VehicleControl ----
        control = carla.VehicleControl()

        # Throttle / brake split
        if accel_cmd >= 0.0:
            control.throttle = min(float(accel_cmd), 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(float(-accel_cmd), 1.0)

        # Steer clamp to [-1, 1]
        control.steer = float(max(-1.0, min(1.0, steer_cmd)))

        # Safety defaults
        control.hand_brake = False
        control.manual_gear_shift = False
        control.reverse = False

        return control

    def compute_steering(self, target_d: float) -> float:
        """
        Compute ONLY the lateral (steering) control for a given lane offset.

        This is used when longitudinal control is handled directly by the RL
        agent (acceleration action) rather than by the speed PID.  The lateral
        PID still tracks the heading error toward a waypoint generated from
        the Frenet offset ``target_d``.

        Args:
            target_d: Desired lateral offset from the reference path center
                      in meters.  Positive = left of reference path.

        Returns:
            Steering command in [-1, 1].
        """
        if carla is None:
            raise RuntimeError(
                "CARLA Python API is not installed.  Cannot compute steering."
            )

        transform = self.vehicle.get_transform()
        current_speed = self.get_speed()
        current_yaw = math.radians(transform.rotation.yaw)

        # Generate a target waypoint in world coordinates
        look_ahead = max(_MIN_LOOK_AHEAD_M, current_speed * _LOOK_AHEAD_TIME_S)
        target_s = self._current_s + look_ahead
        target_s = min(target_s, self._max_s - _SPLINE_END_MARGIN_M)
        target_s = max(0.0, target_s)

        try:
            wp_x, wp_y, _wp_z, _wp_yaw = frenet_to_inertial(
                target_s, target_d, self.global_csp
            )
        except ValueError:
            logger.warning(
                "frenet_to_inertial failed at s=%.1f, d=%.2f; "
                "falling back to straight-ahead waypoint.",
                target_s,
                target_d,
            )
            wp_x = transform.location.x + look_ahead * math.cos(current_yaw)
            wp_y = transform.location.y + look_ahead * math.sin(current_yaw)

        # Heading-error PID
        dx = wp_x - transform.location.x
        dy = wp_y - transform.location.y
        dist_to_wp = math.sqrt(dx * dx + dy * dy)

        if dist_to_wp < 1e-4:
            heading_error = 0.0
        else:
            target_yaw = math.atan2(dy, dx)
            heading_error = self._normalize_angle(target_yaw - current_yaw)

        steer_cmd = self.lat_pid.step(heading_error)
        return float(max(-1.0, min(1.0, steer_cmd)))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
