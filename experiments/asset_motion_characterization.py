"""Phase 5-6: target-motion-only characterization of the asset motion model study.

No attacker/defender dynamics involved -- obstacles disabled so the soldier's OWN proposed
motion is never obstacle-rejected (isolating the motion model itself); only `_move_soldier()` is
exercised, through the real `SoldierEnv` so the RNG stream and boundary/obstacle code paths are
exactly what the simulation uses. No learned policy, no training.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

OUT = Path("results/asset_motion_model_study")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = range(0, 20)
STEPS = 400  # 200 s at dt=0.5
TAUS = (0.5, 1.0, 2.0, 4.0)


def trace(model: str, tau: float, seed: int) -> dict:
    cfg = dataclasses.replace(EnvConfig(), obstacles_enabled=False, asset_motion_model=model, asset_motion_tau_s=tau)
    env = SoldierEnv(cfg)
    env.reset(seed=seed)
    positions = [env._soldier_pos.copy()]
    boundary_reflections = 0
    prev_pos = positions[0]
    for _ in range(STEPS):
        env.step(np.zeros(3, dtype=np.float32))
        new_pos = env._soldier_pos.copy()
        # a boundary reflection is any step whose displacement direction flips sign on an axis
        # where the position is at/near a wall this step (approximate, diagnostic only)
        if abs(new_pos[0]) >= cfg.L - 1e-3 or abs(new_pos[1]) >= cfg.L - 1e-3:
            boundary_reflections += 1
        positions.append(new_pos)
        prev_pos = new_pos
    return {"model": model, "tau": tau, "seed": seed, "positions": [p.tolist() for p in positions],
            "boundary_reflections": boundary_reflections}


def stats(rec: dict, dt: float) -> dict:
    pos = np.array(rec["positions"])
    vel = np.diff(pos, axis=0) / dt  # effective velocity (finite difference), same measure for both models
    speed = np.linalg.norm(vel, axis=1)
    accel = np.diff(vel, axis=0) / dt
    accel_mag = np.linalg.norm(accel, axis=1)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    dheading = np.abs(((np.diff(heading) + np.pi) % (2 * np.pi)) - np.pi)
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    path_length = float(disp.sum())
    net_displacement = float(np.linalg.norm(pos[-1] - pos[0]))

    def autocorr(x, lag):
        x = x - x.mean(axis=0)
        num = np.sum(x[:-lag] * x[lag:]) if lag < len(x) else 0.0
        den = np.sum(x * x)
        return float(num / den) if den > 1e-12 else 0.0

    return {
        "mean_speed": float(speed.mean()), "median_speed": float(np.median(speed)),
        "std_speed": float(speed.std()), "p95_speed": float(np.percentile(speed, 95)),
        "mean_accel": float(accel_mag.mean()), "p95_accel": float(np.percentile(accel_mag, 95)),
        "mean_heading_change_deg": float(np.degrees(dheading.mean())),
        "p95_heading_change_deg": float(np.degrees(np.percentile(dheading, 95))),
        "lag1_autocorr_vx": autocorr(vel[:, 0], 1), "lag1_autocorr_vy": autocorr(vel[:, 1], 1),
        "lag2_autocorr_vx": autocorr(vel[:, 0], 2), "lag2_autocorr_vy": autocorr(vel[:, 1], 2),
        "displacement_rms": float(np.sqrt(np.mean(disp ** 2))),
        "path_length": path_length, "net_displacement": net_displacement,
        "boundary_reflections": rec["boundary_reflections"],
    }


if __name__ == "__main__":
    dt = EnvConfig().dt
    settings = [("legacy_gaussian_random_walk", 1.0)] + [("gauss_markov_velocity", t) for t in TAUS]
    all_stats = {}
    for model, tau in settings:
        key = model if model == "legacy_gaussian_random_walk" else f"gauss_markov_tau{tau}"
        per_seed = [stats(trace(model, tau, seed), dt) for seed in SEEDS]
        agg = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0] if k != "boundary_reflections"}
        agg["boundary_reflections_total"] = int(sum(s["boundary_reflections"] for s in per_seed))
        all_stats[key] = agg
        print(key, "rho" if tau else "", {k: round(v, 4) for k, v in agg.items()})
    json.dump(all_stats, open(OUT / "target_motion_stats.json", "w"), indent=1)
    print("\nwrote", OUT / "target_motion_stats.json")
