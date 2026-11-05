"""10,000-scenario bias audit for the viability-conditioned clutter benchmark (journal_cbf_attacker,
obstacle_cbf_v2 / Prompt 3, brief section 6). Compares the FIRST-DRAW (unconditioned, viability_conditioned_init
=False) attacker state against the ACCEPTED (viability_conditioned_init=True) state for the SAME seed, to quantify
how much conditioning the initial attacker state on braking feasibility shifts the scenario distribution. The
obstacle field and asset initialization are identical between the two runs for a given seed (both use the SAME
_rng_obstacles/_rng_soldier streams, untouched by attacker resampling) -- confirmed directly below as well as by
the dedicated unit test.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from experiments import journal_v2_validation_core as VC
from uav_defend.envs.soldier_env import SoldierEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = "docs/journal_cbf_attacker/obstacle_cbf_v2/"

BIAS_AUDIT_SEEDS = (600_000, 609_999)  # 10,000 scenarios, a fresh block disjoint from every other namespace


def clutter_config() -> object:
    cfg2, _ = VC.build_config("C2", "V2", "passive")
    return dataclasses.replace(cfg2, intercept_radius=1e-6, dynamic_obstacles_enabled=False)


def run_bias_audit(seeds: range) -> dict:
    cfg = clutter_config()
    env_off = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=False)
    env_on = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True)

    n = 0
    accepted_first_draw = 0
    resample_counts = []
    range_before, range_after = [], []
    speed_before, speed_after = [], []
    heading_before, heading_after = [], []
    obstacle_counts = []
    nearest_obstacle_before, nearest_obstacle_after = [], []
    obstacle_field_mismatches = 0
    scenario_generation_failures = 0

    for s in seeds:
        n += 1
        _, info_off = env_off.reset(seed=s)
        _, info_on = env_on.reset(seed=s)

        l1, l2 = env_off._obstacle_layout.obstacles, env_on._obstacle_layout.obstacles
        if len(l1) != len(l2) or not all(np.allclose(a.min_corner, b.min_corner) and np.allclose(a.max_corner, b.max_corner) for a, b in zip(l1, l2)):
            obstacle_field_mismatches += 1
        if not np.allclose(info_off["soldier_pos"], info_on["soldier_pos"]):
            obstacle_field_mismatches += 1  # reused counter: any non-attacker-state mismatch is a protocol violation

        resample_count = int(info_on["viability_resample_count"])
        resample_counts.append(resample_count)
        if resample_count == 0:
            accepted_first_draw += 1
        if info_on["scenario_generation_failure"]:
            scenario_generation_failures += 1

        pos_off = np.asarray(info_off["enemy_pos"], dtype=np.float64)
        vel_off = np.asarray(info_off["enemy_vel"], dtype=np.float64)
        pos_on = np.asarray(info_on["enemy_pos"], dtype=np.float64)
        vel_on = np.asarray(info_on["enemy_vel"], dtype=np.float64)
        asset = np.asarray(info_off["soldier_pos"], dtype=np.float64)

        range_before.append(float(np.linalg.norm(asset - pos_off)))
        range_after.append(float(np.linalg.norm(asset - pos_on)))
        speed_before.append(float(np.linalg.norm(vel_off)))
        speed_after.append(float(np.linalg.norm(vel_on)))
        heading_before.append(float(np.degrees(np.arctan2(vel_off[1], vel_off[0]))))
        heading_after.append(float(np.degrees(np.arctan2(vel_on[1], vel_on[0]))))
        obstacle_counts.append(len(l2))
        nearest_obstacle_before.append(float(env_off._obstacle_layout.min_clearance(pos_off)) if len(l1) else float("inf"))
        nearest_obstacle_after.append(float(env_on._obstacle_layout.min_clearance(pos_on)) if len(l2) else float("inf"))

    def stats(arr):
        a = np.array([x for x in arr if np.isfinite(x)], dtype=np.float64)
        if a.size == 0:
            return {"mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
        return {"mean": float(a.mean()), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
                "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}

    resample_arr = np.array(resample_counts)
    return {
        "n_scenarios": n,
        "accepted_first_draw_fraction": accepted_first_draw / n,
        "required_resampling_fraction": 1.0 - accepted_first_draw / n,
        "mean_resamples": float(resample_arr.mean()), "max_resamples": int(resample_arr.max()),
        "scenario_generation_failures": scenario_generation_failures,
        "obstacle_field_or_asset_mismatches": obstacle_field_mismatches,
        "attacker_range_before": stats(range_before), "attacker_range_after": stats(range_after),
        "attacker_speed_before": stats(speed_before), "attacker_speed_after": stats(speed_after),
        "attacker_heading_deg_before": stats(heading_before), "attacker_heading_deg_after": stats(heading_after),
        "obstacle_count_distribution": {str(k): int((np.array(obstacle_counts) == k).sum()) for k in sorted(set(obstacle_counts))},
        "nearest_obstacle_distance_before": stats(nearest_obstacle_before),
        "nearest_obstacle_distance_after": stats(nearest_obstacle_after),
    }


if __name__ == "__main__":
    result = run_bias_audit(range(BIAS_AUDIT_SEEDS[0], BIAS_AUDIT_SEEDS[1] + 1))
    out_path = PROJECT_ROOT / "docs" / "journal_cbf_attacker" / "obstacle_cbf_v2" / "bias_audit.json"
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
