"""Rendering tests: obstacle footprints must be visible and distinguishable
from the protected asset / defender / hostile UAV markers.

Run directly: python tests/test_obstacle_rendering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv


def _frame(cfg: EnvConfig, seed: int = 1) -> np.ndarray:
    env = SoldierEnv(config=cfg, render_mode="rgb_array")
    env.reset(seed=seed)
    frame = env.render()
    return frame


def test_render_shape_and_dtype_unchanged():
    frame = _frame(EnvConfig())
    assert frame.shape == (100, 100, 3)
    assert frame.dtype == np.uint8


def test_no_obstacles_frame_has_no_gray_footprint_pixels():
    frame = _frame(EnvConfig(obstacles_enabled=False))
    # No pixel should show the obstacle-footprint gray shading (a pure gray
    # R==G==B pixel that is neither black background nor a colored entity
    # marker); background must remain all-black outside entity markers.
    non_black = frame[np.any(frame != 0, axis=-1)]
    for pixel in non_black.reshape(-1, 3):
        assert not (pixel[0] == pixel[1] == pixel[2] and pixel[0] != 0), (
            f"unexpected gray pixel {pixel} with obstacles disabled"
        )


def test_obstacle_footprint_visible_and_distinct_from_entities():
    cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 5.0, 2.0),),
    )
    frame = _frame(cfg)
    L = cfg.L
    size = 100
    x_pix = int(round((-10.0 + L) / (2 * L) * (size - 1)))
    y_pix = int(round((0.0 + L) / (2 * L) * (size - 1)))
    obstacle_pixel = frame[y_pix, x_pix]
    # Gray footprint: R == G == B, non-zero (distinct from black background).
    assert obstacle_pixel[0] == obstacle_pixel[1] == obstacle_pixel[2]
    assert obstacle_pixel[0] != 0
    # Distinct from the pure-color entity markers (blue/green/red).
    assert tuple(obstacle_pixel) not in {(0, 0, 255), (0, 255, 0), (255, 0, 0)}


def test_taller_obstacle_renders_darker_shade():
    cfg_short = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 1.0, 5.0, 5.0, 1.0),),  # height 2
    )
    cfg_tall = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 10.0, 5.0, 5.0, 10.0),),  # height 20
    )
    L = cfg_short.L
    size = 100
    x_pix = int(round((-10.0 + L) / (2 * L) * (size - 1)))
    y_pix = int(round((0.0 + L) / (2 * L) * (size - 1)))
    shade_short = _frame(cfg_short)[y_pix, x_pix][0]
    shade_tall = _frame(cfg_tall)[y_pix, x_pix][0]
    assert shade_tall < shade_short


def test_entity_markers_still_render_over_obstacle_footprint():
    # Defender/asset start co-located at the origin; place an obstacle that
    # does not contain the origin but confirm the render call still
    # succeeds and produces the expected non-black entity-marker colors
    # somewhere in the frame.
    cfg = EnvConfig(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((-10.0, 0.0, 2.0, 5.0, 5.0, 2.0),),
    )
    frame = _frame(cfg)
    pixels = frame.reshape(-1, 3)
    assert any(tuple(p) == (0, 0, 255) for p in pixels)  # soldier/asset (blue)
    assert any(tuple(p) == (0, 255, 0) for p in pixels)  # defender (green)
    assert any(tuple(p) == (255, 0, 0) for p in pixels)  # hostile (red)


if __name__ == "__main__":
    import inspect
    module = sys.modules[__name__]
    test_fns = [obj for name, obj in inspect.getmembers(module) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
