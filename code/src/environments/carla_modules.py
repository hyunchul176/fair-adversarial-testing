"""
CARLA Core Infrastructure Module for Adversarial Scenario Generation.

Adapted from the Frenet_revised project (Moghadam et al., UCSC-ASL) for
CARLA 0.9.16 compatibility. Provides the pygame 2D bird's-eye-view rendering
pipeline, synchronous-mode world management, ego vehicle spawning via Frenet
coordinates, and sensor wrappers needed by the ASG training loop.

RESEARCH CONTEXT:
    CARLA cross-validation is critical to strengthen an AAP submission.
    highway-env alone is acceptable (Zhu et al., AAP 2025), but demonstrating
    that adversarial scenarios trained under the aggressiveness constraint
    transfer to a high-fidelity simulator (CARLA) significantly increases
    reviewer confidence.  This module handles the CARLA side of that pipeline.

CARLA 0.9.16 API compatibility notes:
    - set_target_velocity (not set_velocity) is used for teleporting ego
    - sensor.other.collision / sensor.other.obstacle for collision & LOS
    - synchronous_mode + fixed_delta_seconds for deterministic stepping

Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma
de Barcelona (UAB).
Improved by Majid Moghadam - UCSC - ASL.
Adapted for ASG framework, 2026.
Licensed under the MIT License.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import sys
import weakref
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import carla
    from carla import TrafficLightState as tls
except ImportError:
    raise ImportError(
        "Cannot import carla. Make sure CARLA 0.9.16 Python API is installed "
        "and the CARLA egg/wheel is on your PYTHONPATH."
    )

try:
    import pygame
    from pygame.locals import (
        K_COMMA,
        K_DOWN,
        K_ESCAPE,
        K_F1,
        K_LEFT,
        K_PERIOD,
        K_RIGHT,
        K_SLASH,
        K_SPACE,
        K_TAB,
        K_UP,
        K_a,
        K_d,
        K_h,
        K_i,
        K_m,
        K_p,
        K_q,
        K_s,
        K_w,
        KMOD_CTRL,
        KMOD_SHIFT,
    )
except ImportError:
    raise RuntimeError(
        "Cannot import pygame. Install it with: pip install pygame"
    )

from src.environments.cubic_spline_planner import Spline3D, frenet_to_inertial

logger = logging.getLogger(__name__)

# =============================================================================
# -- Constants ----------------------------------------------------------------
# =============================================================================

# Tango Desktop Project color palette
COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

# Module name constants
MODULE_WORLD = "WORLD"
MODULE_HUD = "HUD"
MODULE_INPUT = "INPUT"
MODULE_TRAFFIC = "TRAFFIC"

# Rendering constants
PIXELS_PER_METER = 12
MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0
PIXELS_AHEAD_VEHICLE = 150


# =============================================================================
# -- Utility helpers ----------------------------------------------------------
# =============================================================================

def euclidean_distance(v1: list, v2: list) -> float:
    """Euclidean distance between two coordinate lists."""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def get_actor_display_name(actor: carla.Actor, truncate: int = 250) -> str:
    """Human-readable actor name derived from its type_id."""
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


class Util:
    """Static utility functions for surface blitting and geometry."""

    @staticmethod
    def blits(
        destination_surface: pygame.Surface,
        source_surfaces: tuple,
        rect: Optional[pygame.Rect] = None,
        blend_mode: int = 0,
    ) -> None:
        for surface in source_surfaces:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v: Any) -> float:
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    @staticmethod
    def get_bounding_box(actor: carla.Actor) -> List[carla.Location]:
        bb = actor.trigger_volume.extent
        corners = [
            carla.Location(x=-bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=-bb.y),
        ]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners


# =============================================================================
# -- ModuleManager ------------------------------------------------------------
# =============================================================================


class ModuleManager:
    """
    Central orchestrator for all CARLA visualization / control modules.

    Modules are registered and then started in order.  Each tick, every module
    is ticked (world sim step, HUD update, input parse).  Each render cycle,
    every module that exposes ``render()`` is called.
    """

    def __init__(self) -> None:
        self.modules: List[Any] = []

    def register_module(self, module: Any) -> None:
        """Register a module for lifecycle management."""
        self.modules.append(module)

    def clear_modules(self) -> None:
        del self.modules[:]

    def tick(self) -> None:
        """Advance every registered module by one step."""
        for module in self.modules:
            module.tick()

    def render(self, display: pygame.Surface) -> None:
        """Render every module that has a render method onto *display*."""
        display.fill(COLOR_ALUMINIUM_4)
        for module in self.modules:
            if hasattr(module, "render"):
                module.render(display)
        pygame.display.flip()

    def get_module(self, name: str) -> Any:
        """Return the first module whose ``.name`` matches *name*."""
        for module in self.modules:
            if module.name == name:
                return module
        return None

    def start_modules(self) -> None:
        """Call ``start()`` on every registered module."""
        for module in self.modules:
            module.start()

    def exit_game(self) -> None:
        """Clean up and exit."""
        self.clear_modules()
        pygame.quit()
        sys.exit()


# =============================================================================
# -- ModuleInput --------------------------------------------------------------
# =============================================================================


class ModuleInput:
    """
    Keyboard and mouse input handler.

    Supports:
        TAB          - toggle hero mode
        Mouse Wheel  - zoom in / zoom out
        Mouse Drag   - move map (map mode only)
        W/S/A/D      - throttle / brake / steer
        Q            - toggle reverse
        Space        - hand-brake
        P            - toggle autopilot
        M            - toggle manual transmission
        ,/.          - gear up/down
        F1           - toggle HUD
        I            - toggle actor ids
        H/?          - toggle help
        ESC          - quit
    """

    def __init__(self, name: str, module_manager: ModuleManager) -> None:
        self.module_manager = module_manager
        self.name = name
        self.mouse_pos: Tuple[int, int] = (0, 0)
        self.mouse_offset: List[float] = [0.0, 0.0]
        self.wheel_offset: float = 0.1
        self.wheel_amount: float = 0.025
        self._steer_cache: float = 0.0
        self.control: Optional[carla.VehicleControl] = None
        self._autopilot_enabled: bool = False
        self.tm_port: Optional[int] = None

    def start(self) -> None:
        self.tm_port = self.module_manager.get_module(MODULE_WORLD).tm_port

    def render(self, display: pygame.Surface) -> None:
        pass

    def tick(self) -> None:
        world = self.module_manager.get_module(MODULE_WORLD)
        self.parse_input(world.clock)

    def _parse_events(self) -> None:
        self.mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.module_manager.exit_game()
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    self.module_manager.exit_game()
                elif event.key == K_h or (
                    event.key == K_SLASH
                    and pygame.key.get_mods() & KMOD_SHIFT
                ):
                    module_hud = self.module_manager.get_module(MODULE_HUD)
                    if module_hud is not None:
                        module_hud.help.toggle()
                elif event.key == K_TAB:
                    module_world = self.module_manager.get_module(MODULE_WORLD)
                    module_hud = self.module_manager.get_module(MODULE_HUD)
                    if module_world.hero_actor is None:
                        module_world.select_hero_actor()
                        self.wheel_offset = HERO_DEFAULT_SCALE
                        self.control = carla.VehicleControl()
                        if module_hud is not None:
                            module_hud.notification("Hero Mode")
                    else:
                        self.wheel_offset = MAP_DEFAULT_SCALE
                        self.mouse_offset = [0, 0]
                        self.mouse_pos = [0, 0]
                        module_world.scale_offset = [0, 0]
                        module_world.hero_actor = None
                        if module_hud is not None:
                            module_hud.notification("Map Mode")
                elif event.key == K_F1:
                    module_hud = self.module_manager.get_module(MODULE_HUD)
                    if module_hud is not None:
                        module_hud.show_info = not module_hud.show_info
                elif event.key == K_i:
                    module_hud = self.module_manager.get_module(MODULE_HUD)
                    if module_hud is not None:
                        module_hud.show_actor_ids = not module_hud.show_actor_ids
                elif isinstance(self.control, carla.VehicleControl):
                    if event.key == K_q:
                        self.control.gear = 1 if self.control.reverse else -1
                    elif event.key == K_m:
                        self.control.manual_gear_shift = (
                            not self.control.manual_gear_shift
                        )
                        world = self.module_manager.get_module(MODULE_WORLD)
                        self.control.gear = (
                            world.hero_actor.get_control().gear
                        )
                        module_hud = self.module_manager.get_module(MODULE_HUD)
                        if module_hud is not None:
                            module_hud.notification(
                                "%s Transmission"
                                % (
                                    "Manual"
                                    if self.control.manual_gear_shift
                                    else "Automatic"
                                )
                            )
                    elif (
                        self.control.manual_gear_shift
                        and event.key == K_COMMA
                    ):
                        self.control.gear = max(-1, self.control.gear - 1)
                    elif (
                        self.control.manual_gear_shift
                        and event.key == K_PERIOD
                    ):
                        self.control.gear = self.control.gear + 1
                    elif event.key == K_p:
                        world = self.module_manager.get_module(MODULE_WORLD)
                        if world.hero_actor is not None:
                            self._autopilot_enabled = not self._autopilot_enabled
                            world.hero_actor.set_autopilot(
                                self._autopilot_enabled, self.tm_port
                            )
                            module_hud = self.module_manager.get_module(
                                MODULE_HUD
                            )
                            if module_hud is not None:
                                module_hud.notification(
                                    "Autopilot %s"
                                    % (
                                        "On"
                                        if self._autopilot_enabled
                                        else "Off"
                                    )
                                )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.wheel_offset += self.wheel_amount
                    if self.wheel_offset >= 1.0:
                        self.wheel_offset = 1.0
                elif event.button == 5:
                    self.wheel_offset -= self.wheel_amount
                    if self.wheel_offset <= 0.1:
                        self.wheel_offset = 0.1

    def _parse_keys(self, milliseconds: float) -> None:
        keys = pygame.key.get_pressed()
        self.control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self.control.steer = round(self._steer_cache, 1)
        self.control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self.control.hand_brake = keys[K_SPACE]

    def _parse_mouse(self) -> None:
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            self.mouse_offset[0] += (1.0 / self.wheel_offset) * (
                x - self.mouse_pos[0]
            )
            self.mouse_offset[1] += (1.0 / self.wheel_offset) * (
                y - self.mouse_pos[1]
            )
            self.mouse_pos = (x, y)

    def parse_input(self, clock: pygame.time.Clock) -> None:
        self._parse_events()
        self._parse_mouse()
        if not self._autopilot_enabled:
            if isinstance(self.control, carla.VehicleControl):
                self._parse_keys(clock.get_time())
                self.control.reverse = self.control.gear < 0
            world = self.module_manager.get_module(MODULE_WORLD)
            if world.hero_actor is not None and self.control is not None:
                world.hero_actor.apply_control(self.control)

    @staticmethod
    def _is_quit_shortcut(key: int) -> bool:
        return (key == K_ESCAPE) or (
            key == K_q and pygame.key.get_mods() & KMOD_CTRL
        )


# =============================================================================
# -- FadingText ---------------------------------------------------------------
# =============================================================================


class FadingText:
    """A text label that fades out over a configurable number of seconds."""

    def __init__(
        self,
        font: pygame.font.Font,
        dim: Tuple[int, int],
        pos: Tuple[int, int],
        world: Any,
    ) -> None:
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left: float = 0
        self.surface = pygame.Surface(self.dim)
        self.world = world

    def set_text(
        self,
        text: str,
        color: pygame.Color = COLOR_WHITE,
        seconds: float = 2.0,
    ) -> None:
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill(COLOR_BLACK)
        self.surface.blit(text_texture, (10, 11))

    def tick(self) -> None:
        delta_seconds = 1e-3 * self.world.clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display: pygame.Surface) -> None:
        display.blit(self.surface, self.pos)


# =============================================================================
# -- HelpText -----------------------------------------------------------------
# =============================================================================


_HELP_TEXT = """\
    TAB          : toggle hero mode
    Mouse Wheel  : zoom in / zoom out
    Mouse Drag   : move map (map mode only)

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    F1           : toggle HUD
    I            : toggle actor ids
    H/?          : toggle help
    ESC          : quit
"""


class HelpText:
    """Toggleable translucent help overlay."""

    def __init__(
        self, font: pygame.font.Font, width: int, height: int
    ) -> None:
        lines = _HELP_TEXT.split("\n")
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (
            0.5 * width - 0.5 * self.dim[0],
            0.5 * height - 0.5 * self.dim[1],
        )
        self.seconds_left: float = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill(COLOR_BLACK)
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, COLOR_WHITE)
            self.surface.blit(text_texture, (22, n * 22))
        self._render = False
        self.surface.set_alpha(220)

    def toggle(self) -> None:
        self._render = not self._render

    def render(self, display: pygame.Surface) -> None:
        if self._render:
            display.blit(self.surface, self.pos)


# =============================================================================
# -- ModuleHUD ----------------------------------------------------------------
# =============================================================================


class ModuleHUD:
    """
    Heads-up display showing server/client FPS, hero info, nearby vehicles.
    Also manages the help overlay and fading notification text.
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        module_manager: ModuleManager,
    ) -> None:
        self.module_manager = module_manager
        self.name = name
        self.dim = (width, height)
        self.world = self.module_manager.get_module(MODULE_WORLD)
        self._init_hud_params()
        self._init_data_params()

    def start(self) -> None:
        pass

    def _init_hud_params(self) -> None:
        fonts = [x for x in pygame.font.get_fonts() if "mono" in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._header_font = pygame.font.SysFont("Arial", 14, True)
        self.help = HelpText(pygame.font.Font(mono, 24), *self.dim)
        self._notifications = FadingText(
            pygame.font.Font(pygame.font.get_default_font(), 20),
            (self.dim[0], 40),
            (0, self.dim[1] - 40),
            self.world,
        )

    def _init_data_params(self) -> None:
        self.show_info: bool = True
        self.show_actor_ids: bool = False
        self._info_text: Dict[str, list] = {}

    def notification(self, text: str, seconds: float = 2.0) -> None:
        self._notifications.set_text(text, seconds=seconds)

    def tick(self) -> None:
        self._notifications.tick()

    def add_info(self, module_name: str, info: list) -> None:
        self._info_text[module_name] = info

    def render_vehicles_ids(
        self,
        vehicle_id_surface: pygame.Surface,
        list_actors: list,
        world_to_pixel: Any,
        hero_actor: Optional[carla.Actor],
        hero_transform: Optional[carla.Transform],
    ) -> pygame.Surface:
        vehicle_id_surface.fill(COLOR_BLACK)
        if self.show_actor_ids:
            vehicle_id_surface.set_alpha(150)
            for actor in list_actors:
                x, y = world_to_pixel(actor[1].location)

                angle = 0
                if hero_actor is not None:
                    angle = -hero_transform.rotation.yaw - 90

                color = COLOR_SKY_BLUE_0
                if int(actor[0].attributes.get("number_of_wheels", 4)) == 2:
                    color = COLOR_CHOCOLATE_0
                if actor[0].attributes.get("role_name", "") == "hero":
                    color = COLOR_CHAMELEON_0
                if actor[0].attributes.get("role_name", "") == "adversary":
                    color = COLOR_SCARLET_RED_0

                font_surface = self._header_font.render(
                    str(actor[0].id), True, color
                )
                rotated_font_surface = pygame.transform.rotate(
                    font_surface, angle
                )
                rect = rotated_font_surface.get_rect(center=(x, y))
                vehicle_id_surface.blit(rotated_font_surface, rect)

        return vehicle_id_surface

    def render(self, display: pygame.Surface) -> None:
        if self.show_info:
            info_surface = pygame.Surface((240, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            i = 0
            for module_name, module_info in self._info_text.items():
                if not module_info:
                    continue
                surface = self._header_font.render(
                    module_name, True, COLOR_ALUMINIUM_0
                ).convert_alpha()
                display.blit(surface, (8 + bar_width / 2, 18 * i + v_offset))
                v_offset += 12
                i += 1
                for item in module_info:
                    if v_offset + 18 > self.dim[1]:
                        break
                    if isinstance(item, list):
                        if len(item) > 1:
                            points = [
                                (x + 8, v_offset + 8 + (1.0 - y) * 30)
                                for x, y in enumerate(item)
                            ]
                            pygame.draw.lines(
                                display, (255, 136, 0), False, points, 2
                            )
                        item = None
                    elif isinstance(item, tuple):
                        if isinstance(item[1], bool):
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (6, 6)
                            )
                            pygame.draw.rect(
                                display,
                                COLOR_ALUMINIUM_0,
                                rect,
                                0 if item[1] else 1,
                            )
                        else:
                            rect_border = pygame.Rect(
                                (bar_h_offset, v_offset + 8),
                                (bar_width, 6),
                            )
                            pygame.draw.rect(
                                display, COLOR_ALUMINIUM_0, rect_border, 1
                            )
                            f = (item[1] - item[2]) / (item[3] - item[2])
                            if item[2] < 0.0:
                                rect = pygame.Rect(
                                    (
                                        bar_h_offset + f * (bar_width - 6),
                                        v_offset + 8,
                                    ),
                                    (6, 6),
                                )
                            else:
                                rect = pygame.Rect(
                                    (bar_h_offset, v_offset + 8),
                                    (f * bar_width, 6),
                                )
                            pygame.draw.rect(
                                display, COLOR_ALUMINIUM_0, rect
                            )
                        item = item[0]
                    if item:
                        surface = self._font_mono.render(
                            item, True, COLOR_ALUMINIUM_0
                        ).convert_alpha()
                        display.blit(surface, (8, 18 * i + v_offset))
                    v_offset += 18
                v_offset += 24
        self._notifications.render(display)
        self.help.render(display)


# =============================================================================
# -- TrafficLightSurfaces -----------------------------------------------------
# =============================================================================


class TrafficLightSurfaces:
    """Pre-rendered, rotatable traffic-light icons for the map view."""

    def __init__(self) -> None:
        def make_surface(tl_state: Any) -> pygame.Surface:
            w = 40
            surface = pygame.Surface((w, 3 * w), pygame.SRCALPHA)
            surface.fill(
                COLOR_ALUMINIUM_5 if tl_state != "h" else COLOR_ORANGE_2
            )
            if tl_state != "h":
                hw = int(w / 2)
                off = COLOR_ALUMINIUM_4
                yellow = COLOR_BUTTER_0
                red = COLOR_SCARLET_RED_0
                green = COLOR_CHAMELEON_0
                pygame.draw.circle(
                    surface,
                    red if tl_state == tls.Red else off,
                    (hw, hw),
                    int(0.4 * w),
                )
                pygame.draw.circle(
                    surface,
                    yellow if tl_state == tls.Yellow else off,
                    (hw, w + hw),
                    int(0.4 * w),
                )
                pygame.draw.circle(
                    surface,
                    green if tl_state == tls.Green else off,
                    (hw, 2 * w + hw),
                    int(0.4 * w),
                )
            return pygame.transform.smoothscale(
                surface, (15, 45) if tl_state != "h" else (19, 49)
            )

        self._original_surfaces = {
            "h": make_surface("h"),
            tls.Red: make_surface(tls.Red),
            tls.Yellow: make_surface(tls.Yellow),
            tls.Green: make_surface(tls.Green),
            tls.Off: make_surface(tls.Off),
            tls.Unknown: make_surface(tls.Unknown),
        }
        self.surfaces = dict(self._original_surfaces)

    def rotozoom(self, angle: float, scale: float) -> None:
        for key, surface in self._original_surfaces.items():
            self.surfaces[key] = pygame.transform.rotozoom(
                surface, angle, scale
            )


# =============================================================================
# -- MapImage -----------------------------------------------------------------
# =============================================================================


class MapImage:
    """
    Bird's-eye-view road map rendered from CARLA waypoints.

    On first use the map is generated from the CARLA topology and cached to
    ``road_maps/road_map_<town>.png``.  Subsequent loads read the cached PNG.
    """

    def __init__(
        self,
        carla_world: carla.World,
        carla_map: carla.Map,
        pixels_per_meter: int = PIXELS_PER_METER,
        show_triggers: bool = False,
        show_connections: bool = False,
        show_spawn_points: bool = False,
    ) -> None:
        self._pixels_per_meter = pixels_per_meter
        self.scale: float = 1.0
        self.show_triggers = show_triggers
        self.show_connections = show_connections
        self.show_spawn_points = show_spawn_points

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = (
            max(
                waypoints, key=lambda x: x.transform.location.x
            ).transform.location.x
            + margin
        )
        max_y = (
            max(
                waypoints, key=lambda x: x.transform.location.y
            ).transform.location.y
            + margin
        )
        min_x = (
            min(
                waypoints, key=lambda x: x.transform.location.x
            ).transform.location.x
            - margin
        )
        min_y = (
            min(
                waypoints, key=lambda x: x.transform.location.y
            ).transform.location.y
            - margin
        )

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)
        self.big_map_surface = pygame.Surface(
            (width_in_pixels, width_in_pixels)
        ).convert()

        # Try loading pre-rendered map from multiple search paths
        map_name_lower = carla_map.name.split("/")[-1].lower()
        search_paths = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "road_maps",
                f"road_map_{map_name_lower}.png",
            ),
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "road_maps",
                f"road_map_{map_name_lower}.png",
            ),
        ]

        loaded = False
        for map_file_name in search_paths:
            map_file_name = os.path.normpath(map_file_name)
            if os.path.exists(map_file_name):
                logger.info("Loading cached road map from %s", map_file_name)
                self.big_map_surface = pygame.image.load(map_file_name)
                loaded = True
                break

        if not loaded:
            logger.info(
                "Pre-rendered road map not found. Generating from CARLA "
                "waypoints (this may take a moment)..."
            )
            self.draw_road_map(
                self.big_map_surface,
                carla_world,
                carla_map,
                self.world_to_pixel,
                self.world_to_pixel_width,
            )
            # Save for future use
            save_dir = os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "..",
                    "road_maps",
                )
            )
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir, f"road_map_{map_name_lower}.png"
            )
            pygame.image.save(self.big_map_surface, save_path)
            logger.info("Saved generated road map to %s", save_path)

        self.surface = self.big_map_surface

    def draw_road_map(
        self,
        map_surface: pygame.Surface,
        carla_world: carla.World,
        carla_map: carla.Map,
        world_to_pixel: Any,
        world_to_pixel_width: Any,
    ) -> None:
        """Render road polygons, lane markings, and arrows onto *map_surface*."""
        map_surface.fill(COLOR_ALUMINIUM_4)
        precision = 0.05

        def lane_marking_color_to_tango(
            lane_marking_color: carla.LaneMarkingColor,
        ) -> pygame.Color:
            tango_color = COLOR_BLACK
            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2
            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0
            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0
            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0
            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0
            return tango_color

        def draw_solid_line(
            surface: pygame.Surface,
            color: pygame.Color,
            closed: bool,
            points: list,
            width: int,
        ) -> None:
            if len(points) >= 2:
                pygame.draw.lines(surface, color, closed, points, width)

        def draw_broken_line(
            surface: pygame.Surface,
            color: pygame.Color,
            closed: bool,
            points: list,
            width: int,
        ) -> None:
            broken_lines = [
                x
                for n, x in enumerate(zip(*(iter(points),) * 20))
                if n % 3 == 0
            ]
            for line in broken_lines:
                pygame.draw.lines(surface, color, closed, line, width)

        def get_lane_markings(
            lane_marking_type: carla.LaneMarkingType,
            lane_marking_color: pygame.Color,
            waypoints: list,
            sign: int,
        ) -> list:
            margin = 0.20
            if lane_marking_type == carla.LaneMarkingType.Broken or (
                lane_marking_type == carla.LaneMarkingType.Solid
            ):
                marking_1 = [
                    world_to_pixel(
                        lateral_shift(w.transform, sign * w.lane_width * 0.5)
                    )
                    for w in waypoints
                ]
                return [(lane_marking_type, lane_marking_color, marking_1)]
            elif (
                lane_marking_type == carla.LaneMarkingType.SolidBroken
                or lane_marking_type == carla.LaneMarkingType.BrokenSolid
            ):
                marking_1 = [
                    world_to_pixel(
                        lateral_shift(w.transform, sign * w.lane_width * 0.5)
                    )
                    for w in waypoints
                ]
                marking_2 = [
                    world_to_pixel(
                        lateral_shift(
                            w.transform,
                            sign * (w.lane_width * 0.5 + margin * 2),
                        )
                    )
                    for w in waypoints
                ]
                return [
                    (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                    (
                        carla.LaneMarkingType.Broken,
                        lane_marking_color,
                        marking_2,
                    ),
                ]
            elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                marking = [
                    world_to_pixel(
                        lateral_shift(
                            w.transform,
                            sign * (w.lane_width * 0.5 - margin),
                        )
                    )
                    for w in waypoints
                ]
                return [
                    (carla.LaneMarkingType.Broken, lane_marking_color, marking)
                ]
            elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                marking = [
                    world_to_pixel(
                        lateral_shift(
                            w.transform,
                            sign * ((w.lane_width * 0.5) - margin),
                        )
                    )
                    for w in waypoints
                ]
                return [
                    (carla.LaneMarkingType.Solid, lane_marking_color, marking)
                ]
            return [
                (carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])
            ]

        def draw_lane_marking(
            surface: pygame.Surface,
            waypoints: list,
            is_left: bool,
        ) -> None:
            sign = -1 if is_left else 1
            marking_type = carla.LaneMarkingType.NONE
            previous_marking_type = carla.LaneMarkingType.NONE
            previous_marking_color = carla.LaneMarkingColor.Other
            waypoints_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE

            for sample in waypoints:
                lane_marking = (
                    sample.left_lane_marking
                    if sign < 0
                    else sample.right_lane_marking
                )
                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    markings = get_lane_markings(
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign,
                    )
                    current_lane_marking = marking_type
                    for marking in markings:
                        waypoints_list.append(marking)
                    temp_waypoints = temp_waypoints[-1:]
                else:
                    temp_waypoints.append(sample)
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            last_markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign,
            )
            for marking in last_markings:
                waypoints_list.append(marking)

            for markings in waypoints_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(
                        surface, markings[1], False, markings[2], 2
                    )
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(
                        surface, markings[1], False, markings[2], 2
                    )

        def draw_arrow(
            surface: pygame.Surface,
            transform: carla.Transform,
            color: pygame.Color = COLOR_ALUMINIUM_2,
        ) -> None:
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            end = transform.location
            start = end - 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir
            pygame.draw.lines(
                surface,
                color,
                False,
                [world_to_pixel(x) for x in [start, end]],
                4,
            )
            pygame.draw.lines(
                surface,
                color,
                False,
                [world_to_pixel(x) for x in [left, start, right]],
                4,
            )

        def draw_traffic_signs(
            surface: pygame.Surface,
            font_surface: pygame.Surface,
            actor: carla.Actor,
            color: pygame.Color = COLOR_ALUMINIUM_2,
            trigger_color: pygame.Color = COLOR_PLUM_0,
        ) -> None:
            transform = actor.get_transform()
            waypoint = carla_map.get_waypoint(transform.location)
            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(
                center=(pixel_pos[0], pixel_pos[1])
            )
            surface.blit(font_surface, offset)

            forward_vector = carla.Location(
                waypoint.transform.get_forward_vector()
            )
            left_vector = (
                carla.Location(
                    -forward_vector.y, forward_vector.x, forward_vector.z
                )
                * waypoint.lane_width
                / 2
                * 0.7
            )
            line = [
                (
                    waypoint.transform.location
                    + (forward_vector * 1.5)
                    + left_vector
                ),
                (
                    waypoint.transform.location
                    + (forward_vector * 1.5)
                    - left_vector
                ),
            ]
            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

            if self.show_triggers:
                corners = Util.get_bounding_box(actor)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, trigger_color, True, corners, 2)

        def lateral_shift(
            transform: carla.Transform, shift: float
        ) -> carla.Location:
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology: list, index: int) -> None:
            topology = [x[index] for x in carla_topology]
            topology = sorted(
                topology, key=lambda w: w.transform.location.z
            )
            for waypoint in topology:
                waypoints = [waypoint]
                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break

                # Draw Road
                road_left_side = [
                    lateral_shift(w.transform, -w.lane_width * 0.5)
                    for w in waypoints
                ]
                road_right_side = [
                    lateral_shift(w.transform, w.lane_width * 0.5)
                    for w in waypoints
                ]
                polygon = road_left_side + [
                    x for x in reversed(road_right_side)
                ]
                polygon = [world_to_pixel(x) for x in polygon]
                if len(polygon) > 2:
                    pygame.draw.polygon(
                        map_surface, COLOR_ALUMINIUM_5, polygon, 5
                    )
                    pygame.draw.polygon(
                        map_surface, COLOR_ALUMINIUM_5, polygon
                    )

                # Draw Shoulders and Parkings
                PARKING_COLOR = COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_5
                final_color = SHOULDER_COLOR

                # Right shoulder
                shoulder = []
                for w in waypoints:
                    r = w.get_right_lane()
                    if r is not None and (
                        r.lane_type == carla.LaneType.Shoulder
                        or r.lane_type == carla.LaneType.Parking
                    ):
                        if r.lane_type == carla.LaneType.Parking:
                            final_color = PARKING_COLOR
                        shoulder.append(r)

                shoulder_left_side = [
                    lateral_shift(w.transform, -w.lane_width * 0.5)
                    for w in shoulder
                ]
                shoulder_right_side = [
                    lateral_shift(w.transform, w.lane_width * 0.5)
                    for w in shoulder
                ]
                polygon = shoulder_left_side + [
                    x for x in reversed(shoulder_right_side)
                ]
                polygon = [world_to_pixel(x) for x in polygon]
                if len(polygon) > 2:
                    pygame.draw.polygon(
                        map_surface, final_color, polygon, 5
                    )
                    pygame.draw.polygon(map_surface, final_color, polygon)
                draw_lane_marking(map_surface, shoulder, False)

                # Left shoulder
                final_color = SHOULDER_COLOR
                shoulder = []
                for w in waypoints:
                    r = w.get_left_lane()
                    if r is not None and (
                        r.lane_type == carla.LaneType.Shoulder
                        or r.lane_type == carla.LaneType.Parking
                    ):
                        if r.lane_type == carla.LaneType.Parking:
                            final_color = PARKING_COLOR
                        shoulder.append(r)

                shoulder_left_side = [
                    lateral_shift(w.transform, -w.lane_width * 0.5)
                    for w in shoulder
                ]
                shoulder_right_side = [
                    lateral_shift(w.transform, w.lane_width * 0.5)
                    for w in shoulder
                ]
                polygon = shoulder_left_side + [
                    x for x in reversed(shoulder_right_side)
                ]
                polygon = [world_to_pixel(x) for x in polygon]
                if len(polygon) > 2:
                    pygame.draw.polygon(
                        map_surface, final_color, polygon, 5
                    )
                    pygame.draw.polygon(map_surface, final_color, polygon)
                draw_lane_marking(map_surface, shoulder, True)

                # Lane markings and arrows
                if not waypoint.is_intersection:
                    draw_lane_marking(map_surface, waypoints, True)
                    draw_lane_marking(map_surface, waypoints, False)
                    for n, wp in enumerate(waypoints):
                        if ((n + 1) % 400) == 0:
                            draw_arrow(map_surface, wp.transform)

        topology = carla_map.get_topology()
        draw_topology(topology, 0)
        draw_topology(topology, 1)

        if self.show_spawn_points:
            for sp in carla_map.get_spawn_points():
                draw_arrow(map_surface, sp, color=COLOR_CHOCOLATE_0)

        if self.show_connections:
            dist = 1.5
            to_pixel = lambda wp: world_to_pixel(wp.transform.location)
            for wp in carla_map.generate_waypoints(dist):
                col = (
                    (0, 255, 255) if wp.is_intersection else (0, 255, 0)
                )
                for nxt in wp.next(dist):
                    pygame.draw.line(
                        map_surface, col, to_pixel(wp), to_pixel(nxt), 2
                    )
                if wp.lane_change & carla.LaneChange.Right:
                    r = wp.get_right_lane()
                    if r and r.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(
                            map_surface, col, to_pixel(wp), to_pixel(r), 2
                        )
                if wp.lane_change & carla.LaneChange.Left:
                    left = wp.get_left_lane()
                    if left and left.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(
                            map_surface,
                            col,
                            to_pixel(wp),
                            to_pixel(left),
                            2,
                        )

        # Draw traffic signs
        actors = carla_world.get_actors()
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont("Arial", font_size, True)

        stops = [actor for actor in actors if "stop" in actor.type_id]
        yields_list = [actor for actor in actors if "yield" in actor.type_id]

        stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        stop_font_surface = pygame.transform.scale(
            stop_font_surface,
            (
                stop_font_surface.get_width(),
                stop_font_surface.get_height() * 2,
            ),
        )

        yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        yield_font_surface = pygame.transform.scale(
            yield_font_surface,
            (
                yield_font_surface.get_width(),
                yield_font_surface.get_height() * 2,
            ),
        )

        for ts_stop in stops:
            draw_traffic_signs(
                map_surface,
                stop_font_surface,
                ts_stop,
                trigger_color=COLOR_SCARLET_RED_1,
            )
        for ts_yield in yields_list:
            draw_traffic_signs(
                map_surface,
                yield_font_surface,
                ts_yield,
                trigger_color=COLOR_ORANGE_1,
            )

    def world_to_pixel(
        self, location: carla.Location, offset: Tuple[int, int] = (0, 0)
    ) -> List[int]:
        """Convert a carla.Location to pixel coordinates on the map surface."""
        x = self.scale * self._pixels_per_meter * (
            location.x - self._world_offset[0]
        )
        y = self.scale * self._pixels_per_meter * (
            location.y - self._world_offset[1]
        )
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width: float) -> int:
        """Convert a world-space width (meters) to pixel width."""
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale: float) -> None:
        """Rescale the map surface if *scale* differs from current scale."""
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(
                self.big_map_surface, (width, width)
            )


# =============================================================================
# -- CollisionSensor ----------------------------------------------------------
# =============================================================================


class CollisionSensor:
    """
    CARLA collision sensor attached to *parent_actor*.

    Records every collision event as ``True`` in a history list.
    """

    def __init__(self, parent_actor: carla.Actor) -> None:
        self.sensor: Optional[carla.Actor] = None
        self.history: List[bool] = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent
        )
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )

    def reset(self) -> None:
        """Clear collision history for a new episode."""
        self.history = []

    def get_collision_history(self) -> List[bool]:
        """Return the list of collision flags."""
        return self.history

    def destroy(self) -> None:
        """Destroy the underlying CARLA sensor actor."""
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

    @staticmethod
    def _on_collision(weak_self: weakref.ref, event: Any) -> None:
        self = weak_self()
        if not self:
            return
        self.history.append(True)


# =============================================================================
# -- LineOfSightSensor --------------------------------------------------------
# =============================================================================


class LineOfSightSensor:
    """
    CARLA obstacle (line-of-sight) sensor.

    Uses ``sensor.other.obstacle`` to detect the nearest dynamic actor ahead
    of *parent_actor* within a configurable range.
    """

    def __init__(
        self,
        parent_actor: carla.Actor,
        distance: float = 200.0,
        hit_radius: float = 0.5,
    ) -> None:
        self.sensor: Optional[carla.Actor] = None
        self.distance: Optional[float] = None
        self.vehicle_ahead: Optional[carla.Actor] = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.obstacle")
        bp.set_attribute("distance", str(distance))
        bp.set_attribute("hit_radius", str(hit_radius))
        bp.set_attribute("only_dynamics", "True")
        bp.set_attribute("debug_linetrace", "False")
        bp.set_attribute("sensor_tick", "0.0")
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent
        )
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LineOfSightSensor._on_los(weak_self, event)
        )

    def reset(self) -> None:
        """Clear detection state for a new episode."""
        self.vehicle_ahead = None
        self.distance = None

    def get_los_distance(self) -> Optional[float]:
        """Return distance to the detected vehicle ahead, or None."""
        return self.distance

    def get_vehicle_ahead(self) -> Optional[carla.Actor]:
        """Return the CARLA actor detected ahead, or None."""
        return self.vehicle_ahead

    def destroy(self) -> None:
        """Destroy the underlying CARLA sensor actor."""
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

    @staticmethod
    def _on_los(weak_self: weakref.ref, event: Any) -> None:
        self = weak_self()
        if not self:
            return
        self.vehicle_ahead = event.other_actor
        self.distance = event.distance


# =============================================================================
# -- ModuleWorld --------------------------------------------------------------
# =============================================================================


class ModuleWorld:
    """
    Core CARLA world module: manages connection, synchronous mode, ego
    spawning, sensor attachment, tick stepping, and 2D bird's-eye rendering.

    RESEARCH MOTIVATION:
        The ego vehicle is spawned at a random Frenet (s, d) position on the
        Town04 highway, matching the same coordinate system used for
        adversarial NPC control.  This allows seamless transfer of the
        adversarial policy trained in highway-env.

    Args:
        name:            Module name (should be MODULE_WORLD).
        host:            CARLA server hostname.
        port:            CARLA server port.
        tm_port:         Traffic Manager port.
        timeout:         CARLA client timeout in seconds.
        module_manager:  Parent ModuleManager.
        width:           Display width (pixels).
        height:          Display height (pixels).
        play_mode:       0 = headless, 1 = pygame 2D, 2 = camera.
        dt:              Fixed timestep for synchronous mode (seconds).
    """

    # Lane and route parameters (matching Frenet_revised/config.yaml)
    LANE_WIDTH: float = 3.5
    MAX_S: int = 3000

    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = 8000,
        timeout: float = 10.0,
        module_manager: Optional[ModuleManager] = None,
        width: int = 1280,
        height: int = 720,
        play_mode: int = 1,
        dt: float = 0.1,
        # Accept pre-existing CARLA handles to avoid creating a second
        # connection that conflicts with the environment's connection.
        client: Optional[carla.Client] = None,
        world: Optional[carla.World] = None,
        town_map: Optional[carla.Map] = None,
    ) -> None:
        self.module_manager = module_manager
        self.width = width
        self.height = height
        self.name = name
        self.host = host
        self.port = port
        self._tm_port_request = tm_port
        self.timeout = timeout
        self.play_mode = play_mode
        self.dt = dt

        self.display: Optional[pygame.Surface] = None
        self.traffic_light_surfaces: Optional[TrafficLightSurfaces] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.server_clock: Optional[pygame.time.Clock] = None

        if self.play_mode:
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.width, self.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF,
            )
            pygame.display.set_caption(
                "ASG - CARLA No Rendering Mode Visualizer"
            )
            font = pygame.font.Font(pygame.font.get_default_font(), 20)
            text_surface = font.render("Rendering map...", True, COLOR_WHITE)
            self.display.blit(
                text_surface,
                text_surface.get_rect(
                    center=(self.width / 2, self.height / 2)
                ),
            )
            pygame.display.flip()
            if self.play_mode == 1:
                self.traffic_light_surfaces = TrafficLightSurfaces()
            self.clock = pygame.time.Clock()
            self.server_clock = pygame.time.Clock()

        # CARLA connection
        self.client: Optional[carla.Client] = None
        self.tm_port: Optional[int] = None
        self.server_fps: float = 0.0
        self.simulation_time: float = 0.0

        # Track whether world/connection was provided externally.
        # When True, start() skips _apply_world_settings() and _spawn_hero()
        # because the owning environment already configured the world and
        # manages its own ego vehicle.  tick() also skips world.tick() and
        # collision_sensor access.
        self._external_world: bool = (world is not None and town_map is not None)

        # World data
        self.world: Optional[carla.World] = None
        self.town_map: Optional[carla.Map] = None
        if self._external_world:
            self.world = world
            self.town_map = town_map
            self.client = client
            if client is not None:
                self.tm_port = client.get_trafficmanager(
                    self._tm_port_request
                ).get_port()
        else:
            self.world, self.town_map = self._get_data_from_carla()
        self.actors_with_transforms: List[Tuple[carla.Actor, carla.Transform]] = []

        # Stored module references (populated in start())
        self.module_hud: Optional[ModuleHUD] = None
        self.module_input: Optional[ModuleInput] = None

        # Surface state
        self.surface_size = [0, 0]
        self.prev_scaled_size: int = 0
        self.scaled_size: int = 0

        # Hero actor
        self.hero_actor: Optional[carla.Actor] = None
        self.hero_transform: Optional[carla.Transform] = None

        # Sensors
        self.collision_sensor: Optional[CollisionSensor] = None
        self.los_sensor: Optional[LineOfSightSensor] = None
        self.collision_hist: List[bool] = []

        self.scale_offset: List[float] = [0, 0]

        # Rendering surfaces
        self.vehicle_id_surface: Optional[pygame.Surface] = None
        self.result_surface: Optional[pygame.Surface] = None
        self.affected_traffic_light: Optional[carla.Actor] = None
        self.map_image: Optional[MapImage] = None
        self.border_round_surface: Optional[pygame.Surface] = None
        self.original_surface_size: Optional[int] = None
        self.hero_surface: Optional[pygame.Surface] = None
        self.actors_surface: Optional[pygame.Surface] = None
        self.initSettings: Optional[carla.WorldSettings] = None
        self.frame: Optional[int] = None

        # Debug / visualization
        self.points_to_draw: Dict[str, Any] = {}

        # Frenet route
        self.global_csp: Optional[Spline3D] = None
        self.init_s: float = 50.0
        self.init_d: float = 0.0 * self.LANE_WIDTH
        self.max_s: int = self.MAX_S
        self.track_length: int = 750

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def update_global_route_csp(self, global_route_csp: Spline3D) -> None:
        """Set the global cubic-spline path used for Frenet spawning."""
        self.global_csp = global_route_csp

    def get_collision_history(self) -> List[bool]:
        return self.collision_hist

    # -----------------------------------------------------------------
    # CARLA connection and configuration
    # -----------------------------------------------------------------

    def _get_data_from_carla(
        self,
    ) -> Tuple[carla.World, carla.Map]:
        """Connect to CARLA, load Town04, and return (world, map)."""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.tm_port = self.client.get_trafficmanager(
                self._tm_port_request
            ).get_port()
            world = self.client.load_world("Town04")
            world.set_weather(
                getattr(carla.WeatherParameters, "ClearNoon")
            )
            logger.info("Map: Town04 --- Weather: ClearNoon")
            town_map = world.get_map()
            return world, town_map
        except RuntimeError as ex:
            logger.error("Failed to connect to CARLA: %s", ex)
            if self.module_manager is not None:
                self.module_manager.exit_game()
            raise

    def _apply_world_settings(
        self,
        synchronous: bool = True,
        no_rendering: bool = False,
        time_step: Optional[float] = None,
    ) -> None:
        """Apply synchronous mode and timestep settings."""
        self.world.tick()
        self.initSettings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous
        settings.no_rendering_mode = no_rendering
        settings.fixed_delta_seconds = time_step
        self.world.apply_settings(settings)
        logger.info(
            "World settings applied: sync=%s, no_render=%s, dt=%s",
            synchronous,
            no_rendering,
            time_step,
        )

    def _recover_config(self) -> None:
        """Restore CARLA world to initial settings."""
        if self.initSettings is not None:
            self.world.apply_settings(self.initSettings)

    # -----------------------------------------------------------------
    # Module lifecycle
    # -----------------------------------------------------------------

    def start(self) -> None:
        """
        Initialize world settings, build map image, spawn hero, and set up
        rendering surfaces.

        When ``_external_world`` is True the owning environment has already
        configured synchronous mode and will spawn its own ego vehicle, so
        we skip ``_apply_world_settings()`` and ``_spawn_hero()``.
        """
        if not self._external_world:
            self._apply_world_settings(
                synchronous=True, no_rendering=False, time_step=self.dt
            )

        if self.play_mode:
            self.module_hud = self.module_manager.get_module(MODULE_HUD)
            self.module_input = self.module_manager.get_module(MODULE_INPUT)

        if self.play_mode == 1:
            self.map_image = MapImage(
                carla_world=self.world,
                carla_map=self.town_map,
                pixels_per_meter=PIXELS_PER_METER,
                show_triggers=False,
                show_connections=False,
                show_spawn_points=False,
            )
            self.original_surface_size = min(self.width, self.height)
            self.surface_size = self.map_image.big_map_surface.get_width()
            self.scaled_size = int(self.surface_size)
            self.prev_scaled_size = int(self.surface_size)

            # Actor rendering surface
            self.actors_surface = pygame.Surface(
                (
                    self.map_image.surface.get_width(),
                    self.map_image.surface.get_height(),
                )
            )
            self.actors_surface.set_colorkey(COLOR_BLACK)
            self.actors_surface.fill(COLOR_BLACK)

            # Vehicle ID surface
            self.vehicle_id_surface = pygame.Surface(
                (self.surface_size, self.surface_size)
            ).convert()
            self.vehicle_id_surface.set_colorkey(COLOR_BLACK)

            # Border circle surface
            self.border_round_surface = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA
            ).convert()
            self.border_round_surface.set_colorkey(COLOR_WHITE)
            self.border_round_surface.fill(COLOR_BLACK)
            center_offset = (int(self.width / 2), int(self.height / 2))
            pygame.draw.circle(
                self.border_round_surface,
                COLOR_ALUMINIUM_1,
                center_offset,
                int(self.height / 2),
            )
            pygame.draw.circle(
                self.border_round_surface,
                COLOR_WHITE,
                center_offset,
                int((self.height - 8) / 2),
            )

            scaled_original_size = self.original_surface_size * (1.0 / 0.9)
            self.hero_surface = pygame.Surface(
                (scaled_original_size, scaled_original_size)
            ).convert()

            self.result_surface = pygame.Surface(
                (self.surface_size, self.surface_size)
            ).convert()
            self.result_surface.set_colorkey(COLOR_BLACK)

        # Spawn hero vehicle (skip when an external world manages the ego)
        if not self._external_world:
            self._spawn_hero()
            self.hero_actor.set_autopilot(False, self.tm_port)

        if self.play_mode:
            self.module_input.wheel_offset = HERO_DEFAULT_SCALE
            self.module_input.control = carla.VehicleControl()

        if self.play_mode == 1:
            weak_self = weakref.ref(self)
            self.world.on_tick(
                lambda timestamp: ModuleWorld._on_world_tick(
                    weak_self, timestamp
                )
            )

    def _spawn_hero(self) -> None:
        """
        Spawn the ego vehicle (Tesla Model 3, red) at the current
        (init_s, init_d) Frenet position along the global cubic spline.
        Attach CollisionSensor and LineOfSightSensor.
        """
        if self.global_csp is None:
            raise RuntimeError(
                "global_csp must be set via update_global_route_csp() before "
                "spawning the hero vehicle."
            )

        blueprint = self.world.get_blueprint_library().filter(
            "vehicle.tesla.model3"
        )[0]
        blueprint.set_attribute("role_name", "hero")
        if blueprint.has_attribute("color"):
            blueprint.set_attribute("color", "10,0,0")

        x, y, z, yaw = frenet_to_inertial(
            self.init_s, self.init_d, self.global_csp
        )
        z += 0.1  # slight elevation to avoid ground collision

        spawn_point = carla.Transform(
            location=carla.Location(x=x, y=y, z=z),
            rotation=carla.Rotation(
                pitch=0.0, yaw=math.degrees(yaw), roll=0.0
            ),
        )
        self.hero_actor = self.world.spawn_actor(blueprint, spawn_point)
        self.hero_actor.set_autopilot(False, self.tm_port)
        logger.info("Spawned hero at s=%.1f, d=%.1f => %s", self.init_s, self.init_d, spawn_point)

        self.hero_transform = self.hero_actor.get_transform()

        # Attach sensors
        self.collision_sensor = CollisionSensor(self.hero_actor)
        self.los_sensor = LineOfSightSensor(self.hero_actor)

    def select_hero_actor(self) -> None:
        """Select the hero actor from the world (fallback for TAB toggle)."""
        hero_vehicles = [
            actor
            for actor in self.world.get_actors().filter("vehicle.*")
            if actor.attributes.get("role_name") == "hero"
        ]
        if hero_vehicles:
            self.hero_actor = hero_vehicles[0]
            self.hero_transform = self.hero_actor.get_transform()

    def reset(self) -> None:
        """
        Reset the ego vehicle to a new random Frenet position.

        init_s is sampled uniformly from [50, max_s - track_length - 50]
        so that all episodes have room for a full track_length trajectory.
        init_d is sampled as a random lane index in [-1, 2] times LANE_WIDTH.
        """
        self.collision_sensor.reset()
        self.los_sensor.reset()

        self.init_s = np.random.uniform(
            50, self.max_s - self.track_length - 50
        )
        self.init_d = np.random.randint(-1, 3) * self.LANE_WIDTH

        x, y, z, yaw = frenet_to_inertial(
            self.init_s, self.init_d, self.global_csp
        )
        z += 0.1

        # Zero out velocities -- CARLA 0.9.16 uses set_target_velocity
        try:
            self.hero_actor.set_target_velocity(
                carla.Vector3D(x=0, y=0, z=0)
            )
            self.hero_actor.set_target_angular_velocity(
                carla.Vector3D(x=0, y=0, z=0)
            )
        except AttributeError:
            # Fallback for older CARLA versions
            self.hero_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
            self.hero_actor.set_angular_velocity(
                carla.Vector3D(x=0, y=0, z=0)
            )

        transform = carla.Transform(
            location=carla.Location(x=x, y=y, z=z),
            rotation=carla.Rotation(
                pitch=0.0, yaw=math.degrees(yaw), roll=0.0
            ),
        )
        self.hero_actor.set_transform(transform)
        logger.debug(
            "Reset hero to s=%.1f, d=%.1f", self.init_s, self.init_d
        )

    def tick(self) -> None:
        """
        Advance the simulation by one step.

        Updates actor transforms, hero transform, steps the CARLA world
        (synchronous tick), and refreshes collision history.

        When ``_external_world`` is True the owning environment is
        responsible for calling ``world.tick()`` and managing collision
        sensors, so this method only refreshes actor/hero transforms and
        HUD info for rendering purposes.
        """
        actors = self.world.get_actors()
        self.actors_with_transforms = [
            (actor, actor.get_transform()) for actor in actors
        ]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
        if self.play_mode and self.clock is not None:
            self._update_hud_info(self.clock)

        if not self._external_world:
            self.world.tick()
            self.collision_hist = self.collision_sensor.get_collision_history()

    def _update_hud_info(self, clock: pygame.time.Clock) -> None:
        """Push status text to the HUD module."""
        hero_mode_text = []
        if self.hero_actor is not None:
            hero_speed = self.hero_actor.get_velocity()
            hero_location = self.hero_actor.get_location()
            hero_speed_text = 3.6 * math.sqrt(
                hero_speed.x ** 2 + hero_speed.y ** 2 + hero_speed.z ** 2
            )
            hero_mode_text = [
                "Hero Mode:                 ON",
                "Hero ID:              %7d" % self.hero_actor.id,
                "Hero Vehicle:  %14s"
                % get_actor_display_name(self.hero_actor, truncate=14),
                "Hero Speed:          %3d km/h" % hero_speed_text,
                "Hero location: %3.3f %3.3f"
                % (hero_location.x, hero_location.y),
            ]
        else:
            hero_mode_text = ["Hero Mode:                OFF"]

        self.server_fps = (
            self.server_clock.get_fps() if self.server_clock else 0.0
        )
        self.server_fps = (
            "inf"
            if self.server_fps == float("inf")
            else round(self.server_fps)
        )

        module_info_text = [
            "Server:  % 16s FPS" % self.server_fps,
            "Client:  % 16s FPS" % round(clock.get_fps()),
            "Simulation Time: % 12s"
            % datetime.timedelta(seconds=int(self.simulation_time)),
            "Map Name:          %10s" % self.town_map.name,
        ]

        if self.module_hud is not None:
            self.module_hud.add_info(self.name, module_info_text)
            self.module_hud.add_info("HERO", hero_mode_text)

    @staticmethod
    def _on_world_tick(weak_self: weakref.ref, timestamp: Any) -> None:
        self = weak_self()
        if not self:
            return
        if self.server_clock is not None:
            self.server_clock.tick()
            self.server_fps = self.server_clock.get_fps()
        self.simulation_time = timestamp.elapsed_seconds

    # -----------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------

    def _split_actors(
        self,
    ) -> Tuple[list, list, list, list]:
        """Partition actors_with_transforms into vehicles, traffic lights,
        speed limits, and walkers."""
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if "vehicle" in actor.type_id:
                vehicles.append(actor_with_transform)
            elif "traffic_light" in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif "speed_limit" in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif "walker" in actor.type_id:
                walkers.append(actor_with_transform)

        # Update HUD with nearby vehicles
        info_text = []
        if self.hero_actor is not None and len(vehicles) > 1:
            location = self.hero_transform.location
            vehicle_list = [
                x[0] for x in vehicles if x[0].id != self.hero_actor.id
            ]

            def distance(v: carla.Actor) -> float:
                return location.distance(v.get_location())

            for n, vehicle in enumerate(sorted(vehicle_list, key=distance)):
                if n > 15:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                info_text.append("% 5d %s" % (vehicle.id, vehicle_type))

        if self.module_hud is not None:
            self.module_hud.add_info("NEARBY VEHICLES", info_text)

        return (vehicles, traffic_lights, speed_limits, walkers)

    def _render_traffic_lights(
        self,
        surface: pygame.Surface,
        list_tl: list,
        world_to_pixel: Any,
    ) -> None:
        self.affected_traffic_light = None
        if self.traffic_light_surfaces is None:
            return

        for tl in list_tl:
            pos = world_to_pixel(tl.get_location())

            if self.hero_actor is not None:
                try:
                    tl_t = tl.get_transform()
                    transformed_tv = tl_t.transform(
                        tl.trigger_volume.location
                    )
                    hero_location = self.hero_actor.get_location()
                    d = hero_location.distance(transformed_tv)
                    s = Util.length(
                        tl.trigger_volume.extent
                    ) + Util.length(self.hero_actor.bounding_box.extent)
                    if d <= s:
                        self.affected_traffic_light = tl
                        srf = self.traffic_light_surfaces.surfaces["h"]
                        surface.blit(srf, srf.get_rect(center=pos))
                except Exception:
                    pass

            srf = self.traffic_light_surfaces.surfaces[tl.state]
            surface.blit(srf, srf.get_rect(center=pos))

    def _render_speed_limits(
        self,
        surface: pygame.Surface,
        list_sl: list,
        world_to_pixel: Any,
        world_to_pixel_width: Any,
    ) -> None:
        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont("Arial", font_size)

        for sl in list_sl:
            x, y = world_to_pixel(sl.get_location())
            white_circle_radius = int(radius * 0.75)
            pygame.draw.circle(
                surface, COLOR_SCARLET_RED_1, (x, y), radius
            )
            pygame.draw.circle(
                surface, COLOR_ALUMINIUM_0, (x, y), white_circle_radius
            )

            limit = sl.type_id.split(".")[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)

            if self.hero_actor is not None:
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)
            else:
                surface.blit(font_surface, (x - radius / 2, y - radius / 2))

    def _render_walkers(
        self,
        surface: pygame.Surface,
        list_w: list,
        world_to_pixel: Any,
    ) -> None:
        for w in list_w:
            color = COLOR_PLUM_0
            bb = w[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
            ]
            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(
        self,
        surface: pygame.Surface,
        list_v: list,
        world_to_pixel: Any,
    ) -> None:
        """
        Render vehicle bounding-box polygons.

        Color scheme:
            - Hero vehicle:   COLOR_CHAMELEON_0 (green)
            - NPC vehicles:   COLOR_SKY_BLUE_0  (blue)
            - Adversarial NPC: COLOR_SCARLET_RED_0 (red) -- role_name == 'adversary'
            - Two-wheelers:   COLOR_CHOCOLATE_1
        """
        for v in list_v:
            color = COLOR_SKY_BLUE_0
            if int(v[0].attributes.get("number_of_wheels", 4)) == 2:
                color = COLOR_CHOCOLATE_1
            if v[0].attributes.get("role_name", "") == "hero":
                color = COLOR_CHAMELEON_0
            if v[0].attributes.get("role_name", "") == "adversary":
                color = COLOR_SCARLET_RED_0

            # Bounding-box polygon (arrow-shaped front)
            bb = v[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x - 0.8, y=-bb.y),
                carla.Location(x=bb.x, y=0),
                carla.Location(x=bb.x - 0.8, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=-bb.y),
            ]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.lines(
                surface,
                color,
                False,
                corners,
                int(math.ceil(4.0 * self.map_image.scale)),
            )

    def _render_actors(
        self,
        surface: pygame.Surface,
        vehicles: list,
        traffic_lights: list,
        speed_limits: list,
        walkers: list,
    ) -> None:
        """Render all actor categories plus debug points."""
        self._render_traffic_lights(
            surface,
            [tl[0] for tl in traffic_lights],
            self.map_image.world_to_pixel,
        )
        self._render_speed_limits(
            surface,
            [sl[0] for sl in speed_limits],
            self.map_image.world_to_pixel,
            self.map_image.world_to_pixel_width,
        )
        self._render_points_to_draw(radius=5)
        self._render_vehicles(
            surface, vehicles, self.map_image.world_to_pixel
        )
        self._render_walkers(
            surface, walkers, self.map_image.world_to_pixel
        )

    def _render_points_to_draw(self, radius: int = 7) -> None:
        """Render debug/visualization points stored in ``points_to_draw``."""
        color_map = {
            "COLOR_ORANGE_0": COLOR_ORANGE_0,
            "COLOR_CHAMELEON_0": COLOR_CHAMELEON_0,
            "COLOR_SKY_BLUE_0": COLOR_SKY_BLUE_0,
            "COLOR_SCARLET_RED_0": COLOR_SCARLET_RED_0,
            "COLOR_BUTTER_0": COLOR_BUTTER_0,
            "COLOR_PLUM_0": COLOR_PLUM_0,
            "COLOR_WHITE": COLOR_WHITE,
        }
        for name, val in self.points_to_draw.items():
            if isinstance(val, list):
                location = val[0]
                color_name = val[1]
            else:
                location = val
                color_name = "COLOR_ORANGE_0"
            color = color_map.get(color_name, COLOR_ORANGE_0)
            center = self.map_image.world_to_pixel(location)
            pygame.draw.circle(self.actors_surface, color, center, radius)

    def _clip_surfaces(self, clipping_rect: pygame.Rect) -> None:
        self.actors_surface.set_clip(clipping_rect)
        self.vehicle_id_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

    def _compute_scale(self, scale_factor: float) -> None:
        m = self.module_input.mouse_pos
        px = (m[0] - self.scale_offset[0]) / float(self.prev_scaled_size)
        py = (m[1] - self.scale_offset[1]) / float(self.prev_scaled_size)
        diff_between_scales = (
            (float(self.prev_scaled_size) * px)
            - (float(self.scaled_size) * px),
            (float(self.prev_scaled_size) * py)
            - (float(self.scaled_size) * py),
        )
        self.scale_offset = (
            self.scale_offset[0] + diff_between_scales[0],
            self.scale_offset[1] + diff_between_scales[1],
        )
        self.prev_scaled_size = self.scaled_size
        self.map_image.scale_map(scale_factor)

    def render(self, display: pygame.Surface) -> None:
        """
        Main render method: compose map, actors, and vehicle IDs into
        a hero-centered, rotated bird's-eye view.
        """
        if self.play_mode != 1:
            return
        if self.actors_with_transforms is None:
            return
        if self.map_image is None:
            return

        self.result_surface.fill(COLOR_BLACK)
        vehicles, traffic_lights, speed_limits, walkers = (
            self._split_actors()
        )

        scale_factor = self.module_input.wheel_offset
        self.scaled_size = int(self.map_image.width * scale_factor)
        if self.scaled_size != self.prev_scaled_size:
            self._compute_scale(scale_factor)

        # Render actors
        self.actors_surface.fill(COLOR_BLACK)
        self._render_actors(
            self.actors_surface,
            vehicles,
            traffic_lights,
            speed_limits,
            walkers,
        )

        # Render vehicle IDs
        if self.module_hud is not None:
            self.module_hud.render_vehicles_ids(
                self.vehicle_id_surface,
                vehicles,
                self.map_image.world_to_pixel,
                self.hero_actor,
                self.hero_transform,
            )

        # Compose surfaces
        surfaces = (
            (self.map_image.surface, (0, 0)),
            (self.actors_surface, (0, 0)),
            (self.vehicle_id_surface, (0, 0)),
        )

        angle = (
            0.0
            if self.hero_actor is None
            else self.hero_transform.rotation.yaw + 90.0
        )
        if self.traffic_light_surfaces is not None:
            self.traffic_light_surfaces.rotozoom(
                -angle, self.map_image.scale
            )

        if self.hero_actor is not None:
            # Hero-centered view
            hero_location_screen = self.map_image.world_to_pixel(
                self.hero_transform.location
            )
            hero_front = self.hero_transform.get_forward_vector()
            translation_offset = (
                hero_location_screen[0]
                - self.hero_surface.get_width() / 2
                + hero_front.x * PIXELS_AHEAD_VEHICLE,
                hero_location_screen[1]
                - self.hero_surface.get_height() / 2
                + hero_front.y * PIXELS_AHEAD_VEHICLE,
            )

            clipping_rect = pygame.Rect(
                translation_offset[0],
                translation_offset[1],
                self.hero_surface.get_width(),
                self.hero_surface.get_height(),
            )
            self._clip_surfaces(clipping_rect)
            Util.blits(self.result_surface, surfaces)

            self.border_round_surface.set_clip(clipping_rect)

            self.hero_surface.fill(COLOR_ALUMINIUM_4)
            self.hero_surface.blit(
                self.result_surface,
                (-translation_offset[0], -translation_offset[1]),
            )

            rotated_result_surface = pygame.transform.rotozoom(
                self.hero_surface, angle, 0.9
            ).convert()
            center = (display.get_width() / 2, display.get_height() / 2)
            rotation_pivot = rotated_result_surface.get_rect(center=center)
            display.blit(rotated_result_surface, rotation_pivot)
            display.blit(self.border_round_surface, (0, 0))
        else:
            # Map mode (no hero)
            translation_offset = (
                self.module_input.mouse_offset[0] * scale_factor
                + self.scale_offset[0],
                self.module_input.mouse_offset[1] * scale_factor
                + self.scale_offset[1],
            )
            center_offset = (
                abs(display.get_width() - self.surface_size)
                / 2
                * scale_factor,
                0,
            )
            clipping_rect = pygame.Rect(
                -translation_offset[0] - center_offset[0],
                -translation_offset[1],
                self.module_hud.dim[0] if self.module_hud else self.width,
                self.module_hud.dim[1] if self.module_hud else self.height,
            )
            self._clip_surfaces(clipping_rect)
            Util.blits(self.result_surface, surfaces)
            display.blit(
                self.result_surface,
                (
                    translation_offset[0] + center_offset[0],
                    translation_offset[1],
                ),
            )

    # -----------------------------------------------------------------
    # Coordinate transforms
    # -----------------------------------------------------------------

    def inertial_to_body_frame(
        self, xi: float, yi: float, psi: float
    ) -> np.ndarray:
        """Transform a point from inertial frame to hero body frame."""
        Xi = np.array([xi, yi])
        R_psi_T = np.array(
            [
                [np.cos(psi), np.sin(psi)],
                [-np.sin(psi), np.cos(psi)],
            ]
        )
        Xt = np.array(
            [
                self.hero_actor.get_transform().location.x,
                self.hero_actor.get_transform().location.y,
            ]
        )
        return np.matmul(R_psi_T, Xi - Xt)

    def body_to_inertial_frame(
        self, xb: float, yb: float, psi: float
    ) -> np.ndarray:
        """Transform a point from hero body frame to inertial frame."""
        Xb = np.array([xb, yb])
        R_psi = np.array(
            [
                [np.cos(psi), -np.sin(psi)],
                [np.sin(psi), np.cos(psi)],
            ]
        )
        Xt = np.array(
            [
                self.hero_actor.get_transform().location.x,
                self.hero_actor.get_transform().location.y,
            ]
        )
        return np.matmul(R_psi, Xb) + Xt

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy all vehicle and sensor actors, recover world settings."""
        logger.info("Destroying vehicle and sensor actors...")
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.los_sensor is not None:
            self.los_sensor.destroy()
        if self.world is not None:
            for actor in self.world.get_actors():
                if "vehicle" in actor.type_id or "sensor" in actor.type_id:
                    try:
                        actor.destroy()
                    except Exception:
                        pass
            logger.info("Recovering world initial configuration...")
            self._recover_config()
