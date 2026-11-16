"""
Adversarial CARLA Environment for Scenario Generation

PURPOSE:
    Core Gymnasium environment where an RL agent controls the ego vehicle as an
    adversarial driver model, generating dangerous-but-realistic scenarios against
    surrounding NPC vehicles managed by IDM/PID.

RESEARCH CONTEXT:
    - The ego vehicle IS the adversarial agent. Whether a driver model is applied
      to the ego or an NPC is a matter of perspective — the model itself is identical.
    - Controlling the ego is simpler because CARLA's infrastructure is ego-centric.
    - The aggressiveness constraint (Hellinger distance to NDD) applies to the ego's
      driving behavior, ensuring behavioral realism.
    - AuthSim (T-ITS 2025) and Hao-NatPrior (T-IV 2024) also used ego-centric control.

ARCHITECTURE:
    RL Agent (PPO) ──> action [acceleration, lane_offset] ──> Ego Vehicle
                          │ accel → direct throttle/brake (no speed PID)
                          │ lane_offset → PID lateral control via EgoController
                          ▼
                 CARLA World Tick
                          │
                          ▼
           obs, reward, SSM, aggressiveness features

    NPC vehicles follow IDM car-following + PID lane-keeping (TrafficManager).

OBSERVATION SPACE (21D, flat, normalized to [-1, 1]):
    Ego state (4):
        ego_speed, ego_ax, ego_lane_offset, ego_lane_id
    Target vehicle relative state (8) — core car-following state:
        rel_distance, rel_speed, rel_accel, target_speed, target_ax,
        target_lane_offset, ttc, thw
    Surrounding context (3):
        left_distance, right_distance, rear_distance
    Safety metrics (2):
        drac, distance (raw, redundant for safety margin)
    Goal conditioning (4):
        type_rear_end, type_cut_in, c_level, ego_speed (dup for goal context)

ACTION SPACE:
    Continuous 2D: [acceleration_norm, lane_offset_norm] in [-1, 1]
    Mapped to: acceleration -> [-3.0, 3.0] m/s², lane_offset -> [-3.5, 3.5] m
    acceleration > 0 → throttle, < 0 → brake (no speed PID — direct control)
    lane_offset → PID lateral steering via EgoController.compute_steering()

    WHY ACCELERATION INSTEAD OF TARGET SPEED:
    PID speed control lacks the precision for maintaining 0.5 m/s speed
    differences needed for realistic car-following (HighD DHW median ~30m at
    25 m/s). Direct acceleration gives the agent instantaneous control over
    closing/opening rate, enabling the fine-grained distance maintenance that
    makes adversarial scenarios realistic.

REWARD:
    R_base = w_type * R_type + w_sev * R_severity + w_ssm * R_ssm + R_alive
           + R_near_miss + R_collision
    Collisions are TERMINAL and heavily penalized (R_collision = -3.0).
    Near-misses (TTC in sweet spot without collision) receive a bonus,
    incentivizing sustained dangerous proximity.
    R_alive provides a modest per-step survival bonus (0.1).
    The Lagrangian constraint term (-lambda * (D_Hellinger - eps)) is NOT applied
    here; it is added externally by ConstrainedAdversarialAgent during training.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from src.environments.cubic_spline_planner import (
    Spline3D,
    frenet_to_inertial,
    inertial_to_frenet_approx,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization constants (from HighD 95th-percentile statistics)
# ---------------------------------------------------------------------------
_MAX_SPEED: float = 40.0       # m/s (~144 km/h)
_MAX_ACCEL: float = 5.0        # m/s^2
_MAX_DISTANCE: float = 100.0   # m
_MAX_DELTA_V: float = 30.0     # m/s
_MAX_TTC: float = 10.0         # s
_MAX_THW: float = 10.0         # s
_MAX_DRAC: float = 10.0        # m/s^2
_MAX_LANE_OFFSET: float = 3.5  # m (one lane width)
_MAX_NUM_LANES: float = 3.0    # Town04 highway: ~3 lanes per direction

# Action mapping ranges
_ACCEL_RANGE = (-3.0, 3.0)        # m/s² — Direct acceleration control for precise
                                   # car-following. Avoids PID speed-tracking lag
                                   # that makes 0.5 m/s differences uncontrollable.
_LANE_OFFSET_RANGE = (-3.5, 3.5)  # meters lateral offset range

# Sentinel values for undefined SSM
_TTC_INF: float = 999.0
_THW_INF: float = 999.0


def _try_import_carla():
    """Lazy-import CARLA to allow importing this module without CARLA installed."""
    try:
        import carla
        return carla
    except ImportError as exc:
        raise ImportError(
            "CARLA Python API not found. Install the carla package or add "
            "the CARLA PythonAPI to your PYTHONPATH."
        ) from exc


class AdversarialCarlaEnv(gymnasium.Env):
    """
    Gymnasium environment for adversarial scenario generation in CARLA.

    The RL agent controls the ego vehicle as an adversarial driver model.
    NPC vehicles are background traffic managed by TrafficManager (IDM + PID).
    The goal is to create dangerous interactions (rear-end, cut-in) with
    controllable severity while maintaining behavioral realism.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = 8000,
        dt: float = 0.1,
        max_episode_steps: int = 200,
        n_npc: int = 15,
        render_mode: str = "human",
        scenario_type: Optional[str] = None,
        c_level: Optional[float] = None,
        w_type: float = 1.0,
        w_severity: float = 3.0,
        w_ssm: float = 1.5,
        r_alive: float = 0.1,
        lane_width: float = 3.5,
        max_s: float = 3000.0,
        track_length: float = 750.0,
        road_map_path: str = "road_maps",
    ):
        super().__init__()

        # Configuration
        self._host = host
        self._port = port
        self._tm_port = tm_port
        self._dt = dt
        self._max_episode_steps = max_episode_steps
        self._n_npc = n_npc
        self.render_mode = render_mode
        self._fixed_scenario_type = scenario_type
        self._fixed_c_level = c_level
        self._w_type = w_type
        self._w_severity = w_severity
        self._w_ssm = w_ssm
        self._r_alive = r_alive
        self._lane_width = lane_width
        self._max_s = max_s
        self._track_length = track_length
        self._road_map_path = road_map_path

        # Aggressiveness calculator for Hellinger distance monitoring
        # (Phase 1: logging only; Phase 2: used as constraint)
        self._agg_calculator = None
        self._hellinger_distance = 0.0

        # Episode state
        self._current_scenario_type: str = "rear_end"
        self._current_c_level: float = 0.5
        self._step_count: int = 0
        self._prev_ego_ax: Optional[float] = None
        self._prev_ego_vx_longitudinal: Optional[float] = None
        self._collision_occurred: bool = False
        self._ego_off_road: bool = False
        self._ttc_min_episode: float = _TTC_INF

        # Gymnasium spaces — 21D observation
        # (4 ego + 8 target-relative + 3 surrounding + 2 safety + 4 goal)
        self.observation_space = spaces.Box(
            low=-np.ones(21, dtype=np.float32),
            high=np.ones(21, dtype=np.float32),
            dtype=np.float32,
        )
        # 2D continuous action: [acceleration_norm, lane_offset_norm]
        # acceleration: [-1,1] → [-3.0, 3.0] m/s²  (direct throttle/brake)
        # lane_offset: [-1,1] → [-3.5, 3.5] m      (PID lateral steering)
        self.action_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # CARLA handles (initialized lazily on first reset)
        self._carla = None
        self._client = None
        self._world = None
        self._tm = None
        self._bp_lib = None
        self._map = None
        self._original_settings = None

        self.ego_vehicle = None
        self._ego_collision_sensor = None
        self._ego_collision_history: list = []
        self._npc_actors: list = []
        self._npc_sensors: list = []

        # High-level action controller (initialized after CARLA connection)
        self._ego_controller = None

        # Global route spline (loaded on first init)
        self._global_csp: Optional[Spline3D] = None
        self._ego_s: float = 0.0  # Current ego Frenet s

        # Pygame module manager (for visualization)
        self._module_manager = None
        self._world_module = None
        self._traffic_module = None
        self._initialized = False

    # ------------------------------------------------------------------
    # CARLA initialization
    # ------------------------------------------------------------------
    def _init_carla(self):
        """Connect to CARLA server and configure synchronous mode."""
        self._carla = _try_import_carla()

        logger.info(f"Connecting to CARLA at {self._host}:{self._port}...")
        self._client = self._carla.Client(self._host, self._port)
        self._client.set_timeout(60.0)

        # Load Town04 (highway map) — skip reload if already loaded
        self._world = self._client.get_world()
        current_map = self._world.get_map().name
        if "Town04" not in current_map:
            logger.info(f"Current map is {current_map}, loading Town04...")
            self._world = self._client.load_world("Town04")
        else:
            logger.info(f"Town04 already loaded.")
        self._world.set_weather(
            getattr(self._carla.WeatherParameters, "ClearNoon")
        )
        self._map = self._world.get_map()
        self._bp_lib = self._world.get_blueprint_library()

        # Traffic manager — each CARLA server needs its own TM port
        # Derive from CARLA port to avoid conflicts in multi-server setups
        tm_port = self._port + 6000  # 2000→8000, 2004→8004, 2008→8008
        self._tm = self._client.get_trafficmanager(tm_port)
        self._tm.set_synchronous_mode(True)

        # Synchronous mode
        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._dt
        if self.render_mode == "none":
            settings.no_rendering_mode = True
        self._world.apply_settings(settings)
        self._world.tick()

        # Load global route spline
        self._load_global_route()

        # Initialize pygame visualization modules
        if self.render_mode == "human":
            self._init_pygame_modules()

        # Initialize AggressivenessCalculator for Hellinger distance monitoring
        try:
            import joblib
            gmm_path = os.path.join("models", "gmm_model_highd.pkl")
            scaler_path = os.path.join("models", "scaler_highd.pkl")
            if os.path.exists(gmm_path) and os.path.exists(scaler_path):
                gmm = joblib.load(gmm_path)
                scaler = joblib.load(scaler_path)
                from src.agents.aggressiveness_calculator import AggressivenessCalculator
                self._agg_calculator = AggressivenessCalculator(
                    gmm_model=gmm, scaler=scaler, window_size=50
                )
                logger.info("AggressivenessCalculator initialized for Hellinger monitoring")
        except Exception as e:
            logger.warning(f"Could not init AggressivenessCalculator: {e}")

        self._initialized = True
        logger.info("CARLA initialized: Town04, synchronous mode, "
                     f"dt={self._dt}s, render={self.render_mode}")

    def _load_global_route(self):
        """Load the Town04 global route and fit a Spline3D."""
        route_file = os.path.join(self._road_map_path, "global_route_town04.npy")
        if not os.path.exists(route_file):
            raise FileNotFoundError(
                f"Global route not found at {route_file}. "
                "Copy from Frenet_revised/road_maps/."
            )
        route = np.load(route_file)  # shape: (N, 3) — x, y, z
        self._global_csp = Spline3D(
            route[:, 0].tolist(),
            route[:, 1].tolist(),
            route[:, 2].tolist(),
        )
        logger.info(f"Global route loaded: {len(route)} waypoints, "
                     f"max_s={self._global_csp.s[-1]:.1f}m")

    def _init_pygame_modules(self):
        """Initialize the pygame-based 2D visualization.

        Passes the already-established CARLA client, world, and map to
        ``ModuleWorld`` so that it reuses the existing connection instead of
        creating a second one (which would cause a timeout).
        """
        from src.environments.carla_modules import (
            ModuleManager, ModuleWorld, ModuleHUD, ModuleInput,
            MODULE_WORLD, MODULE_HUD, MODULE_INPUT,
        )

        self._module_manager = ModuleManager()

        self._world_module = ModuleWorld(
            name=MODULE_WORLD,
            host=self._host,
            port=self._port,
            tm_port=self._tm_port,
            timeout=30.0,
            module_manager=self._module_manager,
            width=1280,
            height=720,
            play_mode=1,
            dt=self._dt,
            # Pass existing CARLA handles to avoid a second connection
            client=self._client,
            world=self._world,
            town_map=self._map,
        )
        self._world_module.update_global_route_csp(self._global_csp)

        self._module_manager.register_module(self._world_module)

        hud = ModuleHUD(
            name=MODULE_HUD,
            width=1280,
            height=720,
            module_manager=self._module_manager,
        )
        self._module_manager.register_module(hud)

        inp = ModuleInput(
            name=MODULE_INPUT,
            module_manager=self._module_manager,
        )
        self._module_manager.register_module(inp)

        self._module_manager.start_modules()

    # ------------------------------------------------------------------
    # Vehicle spawning
    # ------------------------------------------------------------------
    def _spawn_ego(self, init_s: float, init_d: float):
        """Spawn the ego vehicle at the given Frenet coordinates.

        Retries with small offsets in s and z if the initial position collides
        with existing geometry.
        """
        bp = self._bp_lib.filter("vehicle.tesla.model3")[0]
        bp.set_attribute("role_name", "hero")
        if bp.has_attribute("color"):
            bp.set_attribute("color", "10,0,0")  # Red

        # Try several candidate positions before giving up
        s_offsets = [0, 10, -10, 20, -20, 40, -40]
        z_offsets = [0.3, 0.8, 1.5]
        for s_off in s_offsets:
            for z_off in z_offsets:
                candidate_s = init_s + s_off
                x, y, z, yaw = frenet_to_inertial(
                    candidate_s, init_d, self._global_csp
                )
                spawn_tf = self._carla.Transform(
                    location=self._carla.Location(x=x, y=y, z=z + z_off),
                    rotation=self._carla.Rotation(
                        pitch=0.0, yaw=math.degrees(yaw), roll=0.0
                    ),
                )
                self.ego_vehicle = self._world.try_spawn_actor(bp, spawn_tf)
                if self.ego_vehicle is not None:
                    init_s = candidate_s
                    break
            if self.ego_vehicle is not None:
                break

        if self.ego_vehicle is None:
            # Last resort: use CARLA recommended spawn points
            spawn_points = self._map.get_spawn_points()
            for sp in spawn_points:
                self.ego_vehicle = self._world.try_spawn_actor(bp, sp)
                if self.ego_vehicle is not None:
                    logger.warning(
                        "Ego spawned at CARLA spawn point (Frenet position unavailable)"
                    )
                    break
            if self.ego_vehicle is None:
                raise RuntimeError("Failed to spawn ego vehicle at any position")

        try:
            self.ego_vehicle.set_autopilot(False, self._tm.get_port())
        except RuntimeError:
            pass  # RPC timeout on unregister — safe for fresh spawn

        # Zero initial velocity
        try:
            self.ego_vehicle.set_target_velocity(
                self._carla.Vector3D(x=0, y=0, z=0)
            )
        except AttributeError:
            self.ego_vehicle.set_velocity(
                self._carla.Vector3D(x=0, y=0, z=0)
            )

        # Attach collision sensor
        self._attach_collision_sensor()

        self._ego_s = init_s
        logger.info(f"Ego spawned at s={init_s:.1f}, d={init_d:.1f}")

    def _attach_collision_sensor(self):
        """Attach a collision sensor to the ego vehicle."""
        bp = self._bp_lib.find("sensor.other.collision")
        self._ego_collision_sensor = self._world.spawn_actor(
            bp, self._carla.Transform(), attach_to=self.ego_vehicle
        )
        self._ego_collision_history = []

        import weakref
        weak_self = weakref.ref(self)

        def on_collision(event, ref=weak_self):
            obj = ref()
            if obj is not None:
                obj._ego_collision_history.append(event)

        self._ego_collision_sensor.listen(on_collision)

    def _spawn_npcs(self, ego_s: float, ego_d: float):
        """Spawn NPC vehicles around the ego using grid-based placement."""
        from src.environments.carla_traffic import TrafficManager

        self._traffic_module = TrafficManager(
            name="TRAFFIC",
            world_module=None,  # We'll set references manually
            n_spawn_cars=self._n_npc,
            lane_width=self._lane_width,
            max_s=self._max_s,
            track_length=self._track_length,
        )
        # Set CARLA references directly (bypassing start() which expects world_module)
        self._traffic_module.world = self._world
        self._traffic_module._world = self._world
        self._traffic_module.tm_port = self._tm.get_port()
        self._traffic_module.update_global_route_csp(self._global_csp)

        # Filter blueprints — only normal passenger cars/SUVs
        # Allowlist of realistic highway vehicles (no bikes, trucks, buses,
        # micro-cars, emergency vehicles, or vans)
        _ALLOWED_VEHICLES = [
            "vehicle.audi.a2", "vehicle.audi.etron", "vehicle.audi.tt",
            "vehicle.bmw.grandtourer",
            "vehicle.chevrolet.impala",
            "vehicle.dodge.charger_2020",
            "vehicle.ford.crown", "vehicle.ford.mustang",
            "vehicle.jeep.wrangler_rubicon",
            "vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020",
            "vehicle.mercedes.coupe", "vehicle.mercedes.coupe_2020",
            "vehicle.mini.cooper_s", "vehicle.mini.cooper_s_2021",
            "vehicle.nissan.patrol", "vehicle.nissan.patrol_2021",
            "vehicle.seat.leon",
            "vehicle.tesla.model3",
            "vehicle.toyota.prius",
        ]
        self._traffic_module._vehicle_blueprints = [
            bp for bp in self._bp_lib.filter("vehicle.*")
            if bp.id in _ALLOWED_VEHICLES
        ]
        if not self._traffic_module._vehicle_blueprints:
            # Fallback: any 4-wheel non-special vehicle
            self._traffic_module._vehicle_blueprints = [
                bp for bp in self._bp_lib.filter("vehicle.*")
                if bp.has_attribute("number_of_wheels")
                and int(bp.get_attribute("number_of_wheels")) == 4
            ]

        # Also set _blueprint_library for any internal use
        self._traffic_module._blueprint_library = self._bp_lib

        # Create a minimal world_module proxy for CruiseControl fallback
        class _WorldProxy:
            pass
        proxy = _WorldProxy()
        proxy.world = self._world
        proxy.town_map = self._map
        self._traffic_module.world_module = proxy

        self._traffic_module.reset(ego_s, ego_d)

        # Targeted NPCs ahead — close enough for car-following practice.
        # First NPC at ~15m (immediate interaction), second at ~40m (HighD DHW median).
        # Speed 1-2 m/s slower than ego initial → ego approaches naturally.
        ego_lane = round(ego_d / self._lane_width)
        import random
        for offset in [15.0, 40.0]:
            target_s = ego_s + offset
            if target_s < self._max_s:
                target_speed = random.uniform(23.5, 24.5)  # ~1-2 m/s slower than ego (25.6)
                result = self._traffic_module.spawn_one_actor(
                    target_s, ego_lane, target_speed
                )
                if result is not None:
                    logger.info(
                        f"Target NPC spawned ahead: s={target_s:.1f}, "
                        f"lane={ego_lane}, speed={target_speed:.1f} m/s"
                    )
                else:
                    logger.warning(
                        f"Failed to spawn target NPC at s={target_s:.1f}, "
                        f"lane={ego_lane}"
                    )

        logger.info(f"Spawned {len(self._traffic_module.actors_batch)} NPCs")

    def _teleport_ego(self, init_s: float, init_d: float):
        """Teleport existing ego vehicle to new Frenet coordinates."""
        x, y, z, yaw = frenet_to_inertial(init_s, init_d, self._global_csp)

        self.ego_vehicle.set_transform(
            self._carla.Transform(
                location=self._carla.Location(x=x, y=y, z=z + 0.3),
                rotation=self._carla.Rotation(
                    pitch=0.0, yaw=math.degrees(yaw), roll=0.0
                ),
            )
        )
        zero_vec = self._carla.Vector3D(x=0, y=0, z=0)
        for method_name in ("set_target_velocity", "set_velocity"):
            fn = getattr(self.ego_vehicle, method_name, None)
            if fn is not None:
                fn(zero_vec)
                break
        for method_name in ("set_target_angular_velocity", "set_angular_velocity"):
            fn = getattr(self.ego_vehicle, method_name, None)
            if fn is not None:
                fn(zero_vec)
                break

        self._ego_s = init_s

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # First-time initialization
        if not self._initialized:
            self._init_carla()

        # Sample scenario parameters
        if self._fixed_scenario_type is not None:
            self._current_scenario_type = self._fixed_scenario_type
        else:
            self._current_scenario_type = self.np_random.choice(
                ["rear_end", "cut_in"]
            )

        if self._fixed_c_level is not None:
            self._current_c_level = self._fixed_c_level
        else:
            self._current_c_level = float(self.np_random.uniform(0.0, 1.0))

        # Random initial position
        init_s = float(self.np_random.uniform(
            50, self._max_s - self._track_length - 50
        ))
        init_d = float(
            self.np_random.integers(-1, 3) * self._lane_width
        )

        # Destroy previous NPCs
        if self._traffic_module is not None:
            self._traffic_module.destroy()
            self._traffic_module = None

        # Clean up ALL orphaned vehicles and sensors every reset.
        # This prevents ghost actors from previous runs/crashes.
        ego_id = self.ego_vehicle.id if self.ego_vehicle is not None else None
        col_id = self._ego_collision_sensor.id if self._ego_collision_sensor is not None else None
        for actor in self._world.get_actors().filter("vehicle.*"):
            if actor.id != ego_id:
                try:
                    actor.destroy()
                except Exception:
                    pass
        for sensor in self._world.get_actors().filter("sensor.*"):
            if sensor.id != col_id:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
        self._world.tick()

        # Spawn or teleport ego
        if self.ego_vehicle is None:
            self._spawn_ego(init_s, init_d)
        else:
            self._ego_collision_history.clear()
            self._teleport_ego(init_s, init_d)

        # Initialize or reset the high-level action controller
        if self._ego_controller is not None:
            self._ego_controller.reset(init_s)
        else:
            from src.environments.ego_controller import EgoController
            self._ego_controller = EgoController(
                vehicle=self.ego_vehicle,
                global_csp=self._global_csp,
                lane_width=self._lane_width,
                dt=self._dt,
            )
            self._ego_controller.reset(init_s)

        # Let physics settle
        self._world.tick()
        self._world.tick()

        # Spawn NPC traffic
        self._spawn_npcs(init_s, init_d)

        # Register ego with TrafficManager so NPCs' IDM treats ego as the
        # leading vehicle on cut-in/close approach. Without this, the
        # forward-only LOS sensor misses adversarial ego approaching from
        # behind, which makes the adversarial agent converge to fast crashes (every episode
        # ends in collision regardless of c_level).
        if self._traffic_module is not None:
            self._traffic_module.set_ego_vehicle(self.ego_vehicle)

        # Let NPCs settle and PID stabilize at highway speed
        # Need enough ticks for vehicles to reach road surface and
        # for PID steering to align with lane direction
        if self._traffic_module is not None:
            for _ in range(10):
                self._world.tick()
                self._traffic_module.tick()  # NPC PID needs ticks to stabilize

        # Give initial velocity to ego (matching highway speed)
        yaw = frenet_to_inertial(init_s, init_d, self._global_csp)[3]
        init_speed = 25.6  # HighD mean speed (NDD-calibrated)
        init_vel = self._carla.Vector3D(
            x=init_speed * math.cos(yaw),
            y=init_speed * math.sin(yaw),
            z=0.0,
        )
        for method_name in ("set_target_velocity", "set_velocity"):
            fn = getattr(self.ego_vehicle, method_name, None)
            if fn is not None:
                fn(init_vel)
                break

        self._world.tick()

        # Sync the pygame world module's hero_actor so the bird's-eye
        # view camera follows the ego vehicle managed by this environment.
        if self._world_module is not None:
            self._world_module.hero_actor = self.ego_vehicle
            self._world_module.hero_transform = self.ego_vehicle.get_transform()

        # Reset episode state
        self._step_count = 0
        self._prev_ego_ax = None
        self._prev_ego_vx_longitudinal = None
        self._collision_occurred = False
        self._ego_off_road = False
        self._npc_collision = False
        self._hellinger_distance = 0.0
        self._prev_npc_vx = None
        self._ttc_min_episode = _TTC_INF
        if self._agg_calculator is not None:
            self._agg_calculator.reset()

        obs = self._get_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # Apply RL action to ego vehicle
        self._apply_ego_action(action)

        # Tick CARLA world
        self._world.tick()

        # Tick NPC traffic
        if self._traffic_module is not None:
            self._traffic_module.tick()

        # Detect and REMOVE crashed NPCs — don't end the episode,
        # just despawn the problematic NPC so it doesn't become
        # an unrealistic obstacle. This keeps the episode clean.
        self._npc_collision = False
        if self._traffic_module is not None and self._step_count > 40:
            to_remove = []
            for i, actor_dict in enumerate(self._traffic_module.actors_batch):
                npc = actor_dict.get("Actor")
                if npc is None or not npc.is_alive:
                    to_remove.append(i)
                    continue
                npc_vel = npc.get_velocity()
                npc_speed = math.sqrt(npc_vel.x**2 + npc_vel.y**2)
                if npc_speed < 0.5:
                    # NPC stopped on highway — crashed or stuck
                    self._npc_collision = True
                    try:
                        # Despawn the crashed NPC
                        sensor = actor_dict.get("Sensor")
                        if sensor is not None:
                            try:
                                sensor.stop()
                                sensor.destroy()
                            except Exception:
                                pass
                        npc.destroy()
                    except Exception:
                        pass
                    to_remove.append(i)
            # Remove despawned entries (reverse to preserve indices)
            for i in sorted(to_remove, reverse=True):
                self._traffic_module.actors_batch.pop(i)

        # Update ego Frenet position
        ego_loc = self.ego_vehicle.get_location()
        self._ego_s, _ = inertial_to_frenet_approx(
            ego_loc.x, ego_loc.y, self._global_csp, self._ego_s
        )

        # Check collision
        if len(self._ego_collision_history) > 0:
            self._collision_occurred = True

        # Check off-road
        ego_wp = self._map.get_waypoint(
            ego_loc, project_to_road=True,
            lane_type=self._carla.LaneType.Driving
        )
        if ego_wp is not None:
            dist_to_road = ego_loc.distance(ego_wp.transform.location)
            self._ego_off_road = dist_to_road > self._lane_width * 1.5
        else:
            self._ego_off_road = True

        # Compute SSM
        ssm = self._compute_ssm()
        self._ttc_min_episode = min(self._ttc_min_episode, ssm["ttc"])

        # Compute reward
        reward, reward_components = self._compute_reward(ssm)

        # Aggressiveness features
        agg_features = self._compute_aggressiveness_features(ssm)

        # Feed features to AggressivenessCalculator for Hellinger monitoring
        if self._agg_calculator is not None:
            self._agg_calculator.feature_buffer.append(agg_features)
            self._hellinger_distance = self._agg_calculator.compute_hellinger_distance()

        # Termination
        terminated, truncated = self._check_termination()

        # Observation
        obs = self._get_observation(ssm=ssm)
        info = self._build_info(ssm, reward_components, agg_features)
        info["hellinger_distance"] = self._hellinger_distance

        # Update previous longitudinal velocity for next step's acceleration calc
        ego_vel = self.ego_vehicle.get_velocity()
        ego_fwd = self.ego_vehicle.get_transform().get_forward_vector()
        self._prev_ego_vx_longitudinal = (
            ego_vel.x * ego_fwd.x + ego_vel.y * ego_fwd.y
        )

        # Render
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the pygame 2D bird's-eye visualization.

        Before ticking the module manager we update the hero actor reference
        and actor list so the rendered frame reflects the current simulation
        state (the world tick has already been performed by ``step()``).
        """
        if self._world_module is not None and self._world_module.display is not None:
            try:
                import pygame
                # Update hero transform so the camera follows the ego
                if self.ego_vehicle is not None:
                    self._world_module.hero_actor = self.ego_vehicle
                    self._world_module.hero_transform = (
                        self.ego_vehicle.get_transform()
                    )
                # Refresh the actors list for rendering
                self._world_module.actors_with_transforms = [
                    (actor, actor.get_transform())
                    for actor in self._world.get_actors()
                ]
                self._module_manager.tick()
                self._module_manager.render(self._world_module.display)
            except Exception:
                pass  # Gracefully handle rendering failures

    def close(self):
        """Clean up CARLA actors and restore settings."""
        if self._traffic_module is not None:
            self._traffic_module.destroy()
            self._traffic_module = None

        if self._ego_collision_sensor is not None:
            try:
                self._ego_collision_sensor.stop()
                self._ego_collision_sensor.destroy()
            except Exception:
                pass
            self._ego_collision_sensor = None

        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
            self.ego_vehicle = None

        if self._world is not None and self._original_settings is not None:
            try:
                self._world.apply_settings(self._original_settings)
            except Exception:
                pass

        if self._module_manager is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._module_manager = None

        self._initialized = False
        logger.info("Environment closed.")

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------
    def _apply_ego_action(self, action: np.ndarray):
        """
        Convert RL action to CARLA VehicleControl.

        Action[0]: acceleration_norm in [-1, 1] → acceleration [-3.0, 3.0] m/s²
                   Directly maps to throttle (accel > 0) or brake (accel < 0).
                   No speed PID — gives the agent direct control over closing rate.
        Action[1]: lane_offset_norm in [-1, 1] → lateral offset [-3.5, 3.5] m
                   Uses EgoController's lateral PID for smooth lane tracking.

        WHY DIRECT ACCELERATION:
            PID speed control introduces lag and imprecision that prevents the
            agent from learning fine-grained car-following. A 0.5 m/s speed
            difference (critical for HighD-realistic 30m headway at 25 m/s)
            is below PID tracking precision. Direct acceleration gives
            instantaneous control over the closing/opening rate.
        """
        action = np.clip(action, -1.0, 1.0)

        # Map actions to physical values
        accel_cmd = float(action[0]) * _ACCEL_RANGE[1]  # [-3.0, 3.0] m/s²
        target_d = float(action[1]) * _LANE_OFFSET_RANGE[1]  # [-3.5, 3.5] m

        # Longitudinal: direct acceleration → throttle/brake (NO speed PID)
        control = self._carla.VehicleControl()
        if accel_cmd >= 0:
            control.throttle = min(accel_cmd / _ACCEL_RANGE[1], 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel_cmd / _ACCEL_RANGE[1], 1.0)

        # Lateral: PID steering via EgoController (only lateral control)
        self._ego_controller.update_s(self._ego_s)
        steer = self._ego_controller.compute_steering(target_d)
        control.steer = steer

        control.hand_brake = False
        control.manual_gear_shift = False
        self.ego_vehicle.apply_control(control)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self, ssm: Optional[dict] = None) -> np.ndarray:
        """
        Build 21D observation vector, all normalized to ~[-1, 1].

        [0-3]:   Ego state (speed, accel, lane_offset, lane_id)
        [4-11]:  Target vehicle relative state (distance, rel_speed, rel_accel,
                 target_speed, target_accel, target_lane_offset, ttc, thw)
        [12-14]: Surrounding context (left, right, rear distances)
        [15-16]: Safety metrics (drac, distance)
        [17-20]: Goal conditioning (type_rear_end, type_cut_in, c_level, ego_speed)

        The observation is structured around the TARGET vehicle (leading NPC
        for rear_end, adjacent lane NPC for cut_in), providing explicit
        relative kinematics that are critical for learning car-following.

        Args:
            ssm: Pre-computed SSM dict. If None, computed internally.
        """
        ego_tf = self.ego_vehicle.get_transform()
        ego_vel = self.ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)

        # Ego acceleration (velocity-based differentiation)
        ego_fwd = ego_tf.get_forward_vector()
        ego_vx_longitudinal = ego_vel.x * ego_fwd.x + ego_vel.y * ego_fwd.y
        if self._prev_ego_vx_longitudinal is not None:
            ego_ax = (ego_vx_longitudinal - self._prev_ego_vx_longitudinal) / self._dt
        else:
            ego_ax = 0.0

        # Ego lane offset (lateral distance from nearest lane center)
        ego_wp = self._map.get_waypoint(
            ego_tf.location, project_to_road=True
        )
        lane_offset = 0.0
        ego_lane_id: int = 0
        if ego_wp is not None:
            wp_loc = ego_wp.transform.location
            dx = ego_tf.location.x - wp_loc.x
            dy = ego_tf.location.y - wp_loc.y
            wp_yaw = math.radians(ego_wp.transform.rotation.yaw)
            lane_offset = -dx * math.sin(wp_yaw) + dy * math.cos(wp_yaw)
            ego_lane_id = abs(ego_wp.lane_id)

        # --- Target vehicle (scenario-dependent) ---
        target_npc, target_dist = self._find_target_vehicle()

        target_speed: float = 0.0
        target_ax: float = 0.0
        target_lane_offset: float = 0.0
        rel_speed: float = 0.0
        rel_accel: float = 0.0

        if target_npc is not None:
            npc_tf = target_npc.get_transform()
            npc_vel = target_npc.get_velocity()
            target_speed = math.sqrt(npc_vel.x ** 2 + npc_vel.y ** 2)

            # Relative speed (positive = approaching): ego speed projected
            # onto the vector toward the NPC, minus NPC speed projected same way
            npc_loc = npc_tf.location
            ego_loc = ego_tf.location
            dx_to_npc = npc_loc.x - ego_loc.x
            dy_to_npc = npc_loc.y - ego_loc.y
            dist_xy = math.sqrt(dx_to_npc ** 2 + dy_to_npc ** 2)
            if dist_xy > 0.01:
                # Unit vector from ego to NPC
                ux = dx_to_npc / dist_xy
                uy = dy_to_npc / dist_xy
                # Closing speed: ego's approach rate minus NPC's
                ego_approach = ego_vel.x * ux + ego_vel.y * uy
                npc_retreat = npc_vel.x * ux + npc_vel.y * uy
                rel_speed = ego_approach - npc_retreat  # positive = closing
            else:
                rel_speed = 0.0

            # NPC longitudinal acceleration
            npc_fwd = npc_tf.get_forward_vector()
            npc_vx_lon_curr = npc_vel.x * npc_fwd.x + npc_vel.y * npc_fwd.y
            if hasattr(self, '_prev_npc_vx') and self._prev_npc_vx is not None:
                target_ax = (npc_vx_lon_curr - self._prev_npc_vx) / self._dt
            self._prev_npc_vx = npc_vx_lon_curr

            # Relative acceleration
            rel_accel = ego_ax - target_ax

            # Target lane offset
            npc_wp = self._map.get_waypoint(
                npc_tf.location, project_to_road=True
            )
            if npc_wp is not None:
                npc_wp_loc = npc_wp.transform.location
                npc_dx = npc_tf.location.x - npc_wp_loc.x
                npc_dy = npc_tf.location.y - npc_wp_loc.y
                npc_wp_yaw = math.radians(npc_wp.transform.rotation.yaw)
                target_lane_offset = (
                    -npc_dx * math.sin(npc_wp_yaw)
                    + npc_dy * math.cos(npc_wp_yaw)
                )
        else:
            target_dist = _MAX_DISTANCE
            self._prev_npc_vx = None

        # --- SSM for observation ---
        if ssm is None:
            ssm = self._compute_ssm()

        # --- Surrounding context: nearest vehicles in other lanes / behind ---
        _, left_dist = self._find_left_lane_npc()
        _, right_dist = self._find_right_lane_npc()
        _, rear_dist = self._find_rear_npc()

        obs = np.array([
            # Ego state (4)
            np.clip(ego_speed / _MAX_SPEED, -1, 1),
            np.clip(ego_ax / _MAX_ACCEL, -1, 1),
            np.clip(lane_offset / _MAX_LANE_OFFSET, -1, 1),
            np.clip(ego_lane_id / _MAX_NUM_LANES, -1, 1),
            # Relative to target vehicle (8) — core car-following state
            np.clip(target_dist / _MAX_DISTANCE, -1, 1),
            np.clip(rel_speed / _MAX_DELTA_V, -1, 1),
            np.clip(rel_accel / _MAX_ACCEL, -1, 1),
            np.clip(target_speed / _MAX_SPEED, -1, 1),
            np.clip(target_ax / _MAX_ACCEL, -1, 1),
            np.clip(target_lane_offset / _MAX_LANE_OFFSET, -1, 1),
            np.clip(ssm["ttc"] / _MAX_TTC, -1, 1),
            np.clip(ssm["thw"] / _MAX_THW, -1, 1),
            # Surrounding context (3)
            np.clip(left_dist / _MAX_DISTANCE, -1, 1),
            np.clip(right_dist / _MAX_DISTANCE, -1, 1),
            np.clip(rear_dist / _MAX_DISTANCE, -1, 1),
            # Safety metrics (2)
            np.clip(ssm["drac"] / _MAX_DRAC, -1, 1),
            np.clip(target_dist / _MAX_DISTANCE, -1, 1),
            # Goal conditioning (4)
            1.0 if self._current_scenario_type == "rear_end" else 0.0,
            1.0 if self._current_scenario_type == "cut_in" else 0.0,
            self._current_c_level,
            np.clip(ego_speed / _MAX_SPEED, -1, 1),  # duplicated for goal context
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # SSM computation
    # ------------------------------------------------------------------
    def _find_nearest_npc(self) -> Tuple[Any, float]:
        """Find the nearest NPC vehicle to the ego."""
        if self._traffic_module is None or not self._traffic_module.actors_batch:
            return None, _MAX_DISTANCE

        ego_loc = self.ego_vehicle.get_location()
        best_npc = None
        best_dist = _MAX_DISTANCE

        for actor_dict in self._traffic_module.actors_batch:
            npc = actor_dict["Actor"]
            if npc is None or not npc.is_alive:
                continue
            dist = ego_loc.distance(npc.get_location())
            if dist < best_dist:
                best_dist = dist
                best_npc = npc

        return best_npc, best_dist

    def _find_leading_npc(self) -> Tuple[Any, float]:
        """Find the NPC directly ahead of the ego in the same lane."""
        if self._traffic_module is None or not self._traffic_module.actors_batch:
            return None, _MAX_DISTANCE

        ego_loc = self.ego_vehicle.get_location()
        ego_tf = self.ego_vehicle.get_transform()
        ego_fwd = ego_tf.get_forward_vector()

        best_npc = None
        best_dist = _MAX_DISTANCE

        for actor_dict in self._traffic_module.actors_batch:
            npc = actor_dict["Actor"]
            if npc is None or not npc.is_alive:
                continue

            npc_loc = npc.get_location()
            dx = npc_loc.x - ego_loc.x
            dy = npc_loc.y - ego_loc.y

            # Check if NPC is ahead (positive dot product with forward)
            dot = dx * ego_fwd.x + dy * ego_fwd.y
            if dot <= 0:
                continue

            # Check lateral proximity (roughly same lane)
            cross = abs(-dx * ego_fwd.y + dy * ego_fwd.x)
            if cross > self._lane_width:
                continue

            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < best_dist:
                best_dist = dist
                best_npc = npc

        return best_npc, best_dist

    def _find_target_vehicle(self) -> Tuple[Any, float]:
        """
        Find the target vehicle for the current scenario.

        The target vehicle is the one the ego should be interacting with:
        - rear_end: the leading vehicle in the same lane (car-following target)
        - cut_in: the nearest vehicle in an adjacent lane (cut-in target)

        Falls back to the leading NPC if no scenario-specific target is found.

        Returns:
            Tuple of (target_npc, distance). target_npc may be None.
        """
        if self._current_scenario_type == "rear_end":
            target, dist = self._find_leading_npc()
            if target is not None:
                return target, dist
        elif self._current_scenario_type == "cut_in":
            target, dist = self._find_adjacent_lane_npc()
            if target is not None:
                return target, dist
        # Default fallback: leading NPC
        return self._find_leading_npc()

    def _find_adjacent_lane_npc(self) -> Tuple[Any, float]:
        """
        Find the nearest NPC in an adjacent lane (for cut-in scenarios).

        For cut-in generation, the ego needs to be aware of vehicles in
        adjacent lanes to time its lane change. This finds the closest
        NPC that is in a lane immediately next to the ego's current lane.

        Returns:
            Tuple of (npc, distance). npc may be None.
        """
        if self._traffic_module is None or not self._traffic_module.actors_batch:
            return None, _MAX_DISTANCE

        ego_loc = self.ego_vehicle.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)
        if ego_wp is None:
            return None, _MAX_DISTANCE

        ego_lane = ego_wp.lane_id
        best_npc = None
        best_dist = _MAX_DISTANCE

        for actor_dict in self._traffic_module.actors_batch:
            npc = actor_dict.get("Actor")
            if npc is None or not npc.is_alive:
                continue
            npc_loc = npc.get_location()
            npc_wp = self._map.get_waypoint(npc_loc)
            if npc_wp is None:
                continue
            # Check if in adjacent lane (lane_id differs by exactly 1)
            if abs(npc_wp.lane_id - ego_lane) == 1:
                dist = ego_loc.distance(npc_loc)
                if dist < best_dist:
                    best_dist = dist
                    best_npc = npc

        return best_npc, best_dist

    def _find_rear_npc(self) -> Tuple[Any, float]:
        """
        Find the nearest NPC behind the ego in any lane.

        Used for surrounding-context observation (rear vehicle awareness).

        Returns:
            Tuple of (npc, distance). npc may be None.
        """
        if self._traffic_module is None or not self._traffic_module.actors_batch:
            return None, _MAX_DISTANCE

        ego_loc = self.ego_vehicle.get_location()
        ego_tf = self.ego_vehicle.get_transform()
        ego_fwd = ego_tf.get_forward_vector()

        best_npc = None
        best_dist = _MAX_DISTANCE

        for actor_dict in self._traffic_module.actors_batch:
            npc = actor_dict.get("Actor")
            if npc is None or not npc.is_alive:
                continue
            npc_loc = npc.get_location()
            dx = npc_loc.x - ego_loc.x
            dy = npc_loc.y - ego_loc.y
            # Behind ego: negative dot product with forward vector
            dot = dx * ego_fwd.x + dy * ego_fwd.y
            if dot >= 0:
                continue
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < best_dist:
                best_dist = dist
                best_npc = npc

        return best_npc, best_dist

    def _find_left_lane_npc(self) -> Tuple[Any, float]:
        """Find nearest NPC specifically in the left adjacent lane."""
        if self._traffic_module is None or not self._traffic_module.actors_batch:
            return None, _MAX_DISTANCE

        ego_loc = self.ego_vehicle.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)
        if ego_wp is None:
            return None, _MAX_DISTANCE

        ego_lane = ego_wp.lane_id
        best_npc = None
        best_dist = _MAX_DISTANCE

        for actor_dict in self._traffic_module.actors_batch:
            npc = actor_dict.get("Actor")
            if npc is None or not npc.is_alive:
                continue
            npc_loc = npc.get_location()
            npc_wp = self._map.get_waypoint(npc_loc)
            if npc_wp is None:
                continue
            # Left lane: lane_id differs by -1 (CARLA convention: negative = driving dir)
            if npc_wp.lane_id == ego_lane - 1:
                dist = ego_loc.distance(npc_loc)
                if dist < best_dist:
                    best_dist = dist
                    best_npc = npc

        return best_npc, best_dist

    def _find_right_lane_npc(self) -> Tuple[Any, float]:
        """Find nearest NPC specifically in the right adjacent lane."""
        if self._traffic_module is None or not self._traffic_module.actors_batch:
            return None, _MAX_DISTANCE

        ego_loc = self.ego_vehicle.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)
        if ego_wp is None:
            return None, _MAX_DISTANCE

        ego_lane = ego_wp.lane_id
        best_npc = None
        best_dist = _MAX_DISTANCE

        for actor_dict in self._traffic_module.actors_batch:
            npc = actor_dict.get("Actor")
            if npc is None or not npc.is_alive:
                continue
            npc_loc = npc.get_location()
            npc_wp = self._map.get_waypoint(npc_loc)
            if npc_wp is None:
                continue
            # Right lane: lane_id differs by +1
            if npc_wp.lane_id == ego_lane + 1:
                dist = ego_loc.distance(npc_loc)
                if dist < best_dist:
                    best_dist = dist
                    best_npc = npc

        return best_npc, best_dist

    def _compute_ssm(self) -> dict:
        """
        Compute Surrogate Safety Measures between ego and target vehicle.

        Uses _find_target_vehicle() which selects the appropriate NPC based
        on the current scenario type (leading vehicle for rear_end, adjacent
        lane vehicle for cut_in).

        Returns dict with ttc, thw, drac, distance, closing_speed.
        """
        leading_npc, distance = self._find_target_vehicle()

        if leading_npc is None or distance > _MAX_DISTANCE:
            return {
                "ttc": _TTC_INF,
                "thw": _THW_INF,
                "drac": 0.0,
                "distance": min(distance, _MAX_DISTANCE),  # Cap to avoid inf
                "closing_speed": 0.0,
            }

        ego_vel = self.ego_vehicle.get_velocity()
        npc_vel = leading_npc.get_velocity()
        ego_speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)

        # Closing speed (positive = approaching)
        ego_tf = self.ego_vehicle.get_transform()
        ego_fwd = ego_tf.get_forward_vector()

        # Project velocity difference onto forward direction
        dvx = ego_vel.x - npc_vel.x
        dvy = ego_vel.y - npc_vel.y
        closing_speed = dvx * ego_fwd.x + dvy * ego_fwd.y

        # TTC
        if closing_speed > 0.01 and distance > 0.1:
            ttc = distance / closing_speed
        else:
            ttc = _TTC_INF

        # THW
        if ego_speed > 0.1:
            thw = distance / ego_speed
        else:
            thw = _THW_INF

        # DRAC
        if distance > 0.1 and closing_speed > 0.01:
            drac = (closing_speed ** 2) / (2.0 * distance)
        else:
            drac = 0.0

        return {
            "ttc": min(ttc, _TTC_INF),
            "thw": min(thw, _THW_INF),
            "drac": drac,
            "distance": distance,
            "closing_speed": closing_speed,
        }

    # ------------------------------------------------------------------
    # Aggressiveness features (6D, matching HighD Phase 0)
    # ------------------------------------------------------------------
    def _compute_aggressiveness_features(self, ssm: dict) -> np.ndarray:
        """
        Extract 6D aggressiveness feature vector from ego's current state.

        Features: [jerk_x, a_x, delta_v, delta_d, thw, ttc]
        These match the HighD feature definition exactly for constraint computation.
        """
        ego_vel = self.ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)

        # Acceleration from velocity-based differentiation
        ego_fwd = self.ego_vehicle.get_transform().get_forward_vector()
        ego_vx_longitudinal = ego_vel.x * ego_fwd.x + ego_vel.y * ego_fwd.y
        if self._prev_ego_vx_longitudinal is not None:
            a_x = (ego_vx_longitudinal - self._prev_ego_vx_longitudinal) / self._dt
        else:
            a_x = 0.0

        # Jerk
        if self._prev_ego_ax is not None:
            jerk_x = (a_x - self._prev_ego_ax) / self._dt
        else:
            jerk_x = 0.0
        self._prev_ego_ax = a_x

        # Relative speed to leading vehicle (positive = approaching)
        delta_v = ssm.get("closing_speed", 0.0)

        # Distance headway
        delta_d = ssm.get("distance", _MAX_DISTANCE)

        # THW and TTC
        thw = ssm.get("thw", _THW_INF)
        ttc = ssm.get("ttc", _TTC_INF)

        return np.array(
            [jerk_x, a_x, delta_v, delta_d, thw, ttc], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, ssm: dict) -> Tuple[float, dict]:
        """
        Compute base adversarial reward R_base.

        R_base = w_type * R_type + w_sev * R_severity + w_ssm * R_ssm + R_alive

        KEY DESIGN PRINCIPLE:
            Collisions are PUNISHED, not rewarded. The goal is to create
            sustained, dangerous-but-realistic interactions (tailgating, close
            following, dangerous cut-ins) that mirror aggressive moments
            observed in HighD naturalistic data. Instant head-on crashes are
            trivially achievable and scientifically meaningless.

        This is the adversarial objective ONLY. The behavioral realism constraint
        (-lambda * (D_Hellinger - eps)) is added externally.
        """
        # R_type: scenario type achievement (sustained interaction, not crash)
        r_type = self._compute_r_type(ssm)

        # R_severity: THW target tracking (replaces TTC-based reward).
        #
        # WHY THW instead of TTC:
        # TTC = distance / relative_speed requires constant approach (Δv > 0)
        # to be finite, making collision structurally inevitable.
        # THW = distance / ego_speed can be maintained at any Δv, including
        # Δv ≈ 0 (speed matching), enabling stable car-following.
        #
        # Severity mapping: THW_target = 3.0 - 2.5 * c_level
        #   c=0.0 → THW=3.0s (conservative)
        #   c=0.5 → THW=1.75s (moderate — HighD THW median is 1.71s)
        #   c=0.8 → THW=1.0s (aggressive — HighD aggressive THW < 1.0s)
        #   c=1.0 → THW=0.5s (extreme)
        thw_target = 3.0 - 2.5 * self._current_c_level
        thw_actual = ssm["thw"]

        ego_vel = self.ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)

        # Speed gate: THW reward is scaled by ego speed.
        # At 20+ m/s: full reward. At 10 m/s: 50%. At 3 m/s: 15%.
        # This prevents "slow crawling" local optimum — only highway-speed
        # car-following earns meaningful positive reward.
        _MIN_REWARD_SPEED = 20.0  # m/s — below this, reward is attenuated
        speed_factor = min(ego_speed / _MIN_REWARD_SPEED, 1.0)

        if ego_speed > 1.0 and ssm["distance"] < _MAX_DISTANCE:
            distance = ssm["distance"]
            proximity_bonus = 0.5 * max(0.0, 1.0 - distance / _MAX_DISTANCE)

            if thw_actual < _MAX_THW:
                thw_error = min(abs(thw_actual - thw_target) / max(thw_target, 0.3), 1.0)
                base_severity = -thw_error + proximity_bonus
                # Bonuses for good behavior
                if abs(thw_actual - thw_target) < 0.3:
                    base_severity += 0.5
                closing_speed = ssm.get("closing_speed", 0.0)
                if abs(closing_speed) < 1.0 and thw_actual < thw_target * 2.0:
                    base_severity += 0.3
                # Apply speed gate: positive rewards scaled by speed,
                # negative rewards kept full (don't reduce penalties for slow driving)
                if base_severity > 0:
                    r_severity = base_severity * speed_factor
                else:
                    r_severity = base_severity
            else:
                r_severity = -1.0 + proximity_bonus
        else:
            r_severity = -1.0

        # R_ssm: physical plausibility penalties (stricter than before)
        r_ssm = 0.0

        # TTC < 0.3s is extremely unrealistic — about to crash unavoidably
        if ssm["ttc"] < 0.3:
            r_ssm -= 3.0

        # Unrealistic deceleration requirement
        if ssm["drac"] > 8.0:
            r_ssm -= 1.0

        # Collision is an UNREALISTIC outcome — punish heavily
        if self._collision_occurred:
            r_ssm -= 2.0

        # Off-road penalty
        if self._ego_off_road:
            r_ssm -= 2.0

        # Speed maintenance: STRONG penalty for slowing on highway.
        # Stopping is the most unrealistic highway behavior and must be
        # worse than approaching (which risks collision).
        # Gradient: starts at 15 m/s, maxes at -2.0 for full stop.
        if ego_speed < 15.0:
            # Linear penalty: 15 m/s → 0, 0 m/s → -2.0
            r_ssm -= 2.0 * (1.0 - ego_speed / 15.0)

        # Near-miss bonus: THW-based, rewarding stable close following
        r_near_miss = 0.0
        if not self._collision_occurred and ego_speed > 1.0:
            if thw_actual < thw_target * 1.2 and thw_actual > 0.3:
                # Within 20% of target THW without collision — good following
                r_near_miss = 0.3

        # Collision penalty (terminal — fires once at episode end).
        # Kept moderate: episode termination + lost future rewards is
        # the main punishment. Explicit penalty just ensures the signal.
        r_collision = 0.0
        if self._collision_occurred:
            r_collision = -3.0

        r_total = (
            self._w_type * r_type
            + self._w_severity * r_severity
            + self._w_ssm * r_ssm
            + r_near_miss
            + r_collision
        )

        components = {
            "r_type": r_type,
            "r_severity": r_severity,
            "r_ssm": r_ssm,
            "r_near_miss": r_near_miss,
            "r_collision": r_collision,
            "r_total": r_total,
            "thw_target": thw_target,
        }

        return r_total, components

    def _compute_r_type(self, ssm: dict) -> float:
        """
        Compute scenario-type achievement reward based on sustained interaction.

        KEY CHANGE from previous version:
            - Collision is now PUNISHED (-0.5), not rewarded (+1.0)
            - Reward is proportional to closeness while MAINTAINING distance
            - Near-misses (TTC < 1.0 but no collision) are valuable
            - The agent learns to be aggressively close, not to crash
        """
        distance = ssm["distance"]
        ttc = ssm["ttc"]

        # Collision penalty applies to ALL scenario types
        if self._collision_occurred:
            return -0.5

        if self._current_scenario_type == "rear_end":
            # Reward for maintaining close following distance (not crashing)
            # Ego must be behind the leading NPC (which _find_leading_npc ensures)
            if distance < 30.0 and ttc > 1.0:
                # Sustained close following — the closer, the higher the reward
                return float(np.clip((30.0 - distance) / 30.0, 0.0, 1.0))
            elif ttc < 1.0 and ttc > 0.0 and not self._collision_occurred:
                # Near-miss situation: TTC < 1.0 but no collision yet
                return 0.5
            elif distance < _MAX_DISTANCE:
                # Far but at least interacting — small reward
                return 0.0
            return -0.5  # No NPC in range — no interaction

        elif self._current_scenario_type == "cut_in":
            nearest_npc, nearest_dist = self._find_target_vehicle()
            if nearest_npc is not None and nearest_dist < 50.0:
                ego_loc = self.ego_vehicle.get_location()
                npc_loc = nearest_npc.get_location()
                ego_wp = self._map.get_waypoint(ego_loc)
                npc_wp = self._map.get_waypoint(npc_loc)
                if ego_wp and npc_wp:
                    same_lane = (ego_wp.lane_id == npc_wp.lane_id)
                    if same_lane and nearest_dist < 15.0:
                        # Phase 2: Successfully cut in — same lane, close range
                        return float(
                            1.0 * np.clip((15.0 - nearest_dist) / 15.0, 0.0, 1.0)
                        )
                    elif not same_lane and nearest_dist < 30.0:
                        # Phase 1: Adjacent lane, positioning for cut-in
                        return float(
                            0.3 * np.clip((30.0 - nearest_dist) / 30.0, 0.0, 1.0)
                        )
                return 0.0
            return -0.5  # No NPC nearby

        return 0.0

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _check_termination(self) -> Tuple[bool, bool]:
        """
        Check episode termination conditions.

        Returns (terminated, truncated).
        """
        terminated = False
        truncated = False

        # Grace period: first 30 steps non-terminal for collision.
        _GRACE_STEPS = 30
        if self._collision_occurred and self._step_count > _GRACE_STEPS:
            terminated = True
        if self._ego_off_road:
            terminated = True
        # NPC anomalies are handled by despawning (not episode termination).
        # self._npc_collision is still logged for monitoring.
        if self._step_count >= self._max_episode_steps:
            truncated = True

        return terminated, truncated

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------
    def _build_info(
        self,
        ssm: Optional[dict] = None,
        reward_components: Optional[dict] = None,
        agg_features: Optional[np.ndarray] = None,
    ) -> dict:
        """Build the info dict returned by reset/step."""
        ego_vel = self.ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)

        # Determine termination reason for logging
        if self._collision_occurred:
            term_reason = "ego_collision"
        elif self._ego_off_road:
            term_reason = "off_road"
        elif self._npc_collision:
            term_reason = "npc_anomaly"
        elif self._step_count >= self._max_episode_steps:
            term_reason = "timeout"
        else:
            term_reason = "running"

        info = {
            "scenario_type": self._current_scenario_type,
            "c_level": self._current_c_level,
            "ego_speed": ego_speed,
            "collision": self._collision_occurred,
            "off_road": self._ego_off_road,
            "npc_collision": self._npc_collision,
            "term_reason": term_reason,
            "step_count": self._step_count,
            "ttc_min_episode": self._ttc_min_episode,
        }

        if ssm is not None:
            info["ttc"] = ssm["ttc"]
            info["thw"] = ssm["thw"]
            info["drac"] = ssm["drac"]
            info["distance"] = ssm["distance"]

        if reward_components is not None:
            info["reward_components"] = reward_components

        if agg_features is not None:
            info["aggressiveness_features"] = agg_features

        return info

    # ------------------------------------------------------------------
    # Frenet helpers
    # ------------------------------------------------------------------
    def get_ego_frenet(self) -> Tuple[float, float]:
        """Get ego vehicle's current Frenet (s, d) coordinates."""
        ego_loc = self.ego_vehicle.get_location()
        return inertial_to_frenet_approx(
            ego_loc.x, ego_loc.y, self._global_csp, self._ego_s
        )
