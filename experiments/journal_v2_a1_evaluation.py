"""Canonical V2 A1 evaluation protocol (`a1_v2_training_v1`): episode runner, matched-A0 comparison,
pre-registered acceptance gates, and representative-episode selection for visual inspection.

This module defines and tests the protocol; it does not run the full evaluation. The eventual full run is

    3 A1 checkpoints x 5 defenders x 4 clutter levels x 100 scenarios (seeds 140000-140099)

plus matched A0 episodes on exactly the same scenario seeds. Numbers are reported, never tuned against.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from experiments.journal_v2_a1 import (
    A1_V2_CLUTTER_LEVELS,
    A1_V2_DEFENDER_FAMILIES,
    A1V2DefenderBank,
    A1V2TrainingEnv,
    evaluation_seeds,
)
from experiments.journal_v2_validation_core import count_overshoots, orbit_revolutions

OBSTACLE_COLLISION_OUTCOMES = ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision")

# Pre-registered acceptance thresholds (a1_v2_training_v1). Changing any of them is a new protocol version.
A1_V2_ACCEPTANCE_THRESHOLDS: dict = {
    "c0_max_endurance_exhaustion_rate": 0.05,        # per A1 seed, C0, pooled over defenders; above -> automatic rejection
    "c0_max_fraction_path_excess_above_2x": 0.05,    # per A1 seed, C0: no persistent circling
    "mission_min_capture_advantage": 0.05,           # A1 - A0 capture rate (pooled over levels, matched seeds)
    "mission_min_seeds_with_advantage": 2,           # ... for at least one defender family, in >= 2 of 3 seeds
    "generalization_max_seed_spread": 0.20,          # max-min pooled capture rate over A1 seeds
    "generalization_max_shortfall_vs_a0": 0.10,      # no A1 seed more than 10 pp below A0 pooled capture
    "obstacle_max_collision_rate": 0.02,             # per A1 seed and clutter level (A0 reference: 0)
    "path_max_fraction_above_2x_given_success": 0.10,
    "path_max_fraction_above_5x_given_success": 0.01,
}
PATH_EXCESS_FLAGS = (1.3, 2.0, 5.0)


# ----------------------------------------------------------------------------- episode runner
def run_episode(attacker, attacker_id: str, level: str, family: str, seed: int,
                env: A1V2TrainingEnv | None = None) -> dict:
    """One deterministic episode. `attacker` is None for the unmodified scripted A0, otherwise any object
    with `predict(observation, deterministic=True) -> (action, state)` (an SB3 PPO model)."""
    if env is None:
        env = A1V2TrainingEnv(0, 0, scripted_a0=attacker is None)
    obs, info = env.reset(options={"scenario": (level, family, int(seed))})
    dt = float(env.soldier_env.config.dt)
    attacker_xyz, asset_xyz, ranges, defender_min, avoid_steps = [], [], [], math.inf, 0
    done = False
    while not done:
        if attacker is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action, _ = attacker.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        attacker_xyz.append(np.asarray(info["enemy_pos"], dtype=np.float64))
        asset_xyz.append(np.asarray(info["soldier_pos"], dtype=np.float64))
        ranges.append(float(info["enemy_soldier_dist"]))
        defender_min = min(defender_min, float(info["defender_enemy_dist"]))
        avoid_steps += int(bool(info.get("enemy_obstacle_avoidance_active")))
    episode = info["a1_episode"]
    orbit_max, _ = orbit_revolutions(np.array(attacker_xyz), np.array(asset_xyz))
    return {
        "attacker": attacker_id, "level": level, "defender": family, "seed": int(seed),
        "outcome": episode["outcome"], "steps": episode["steps"],
        "flight_time_s": float(info.get("attacker_flight_time_s", episode["steps"] * dt)),
        "path_length_m": episode["path_length_m"], "chord_m": episode["chord_m"], "path_excess": episode["path_excess"],
        "initial_range_m": episode["initial_range_m"], "min_defender_separation_m": defender_min,
        "obstacle_collision": episode["outcome"] in OBSTACLE_COLLISION_OUTCOMES,
        "endurance_exhausted": episode["endurance_exhausted"], "captured": episode["outcome"] == "soldier_caught",
        "overshoots": int(count_overshoots(np.array(ranges), dt)), "orbit_revolutions_max": float(orbit_max),
        "avoidance_steps": avoid_steps, "return_undiscounted": episode["return_undiscounted"],
        "reward_components": episode["components"],
    }


def evaluation_plan(attacker_ids: tuple[str, ...] = ("A1-201", "A1-202", "A1-203")) -> list[tuple[str, str, str, int]]:
    """The full frozen plan: (attacker, level, defender, seed) for every A1 checkpoint plus matched A0."""
    return [
        (attacker, level, family, seed)
        for attacker in (*attacker_ids, "A0")
        for level in A1_V2_CLUTTER_LEVELS
        for family in A1_V2_DEFENDER_FAMILIES
        for seed in evaluation_seeds()
    ]


def run_scenarios(attacker, attacker_id: str, scenarios, bank: A1V2DefenderBank | None = None) -> list[dict]:
    env = A1V2TrainingEnv(0, 0, defender_bank=bank, scripted_a0=attacker is None)
    return [run_episode(attacker, attacker_id, lvl, fam, seed, env=env) for (lvl, fam, seed) in scenarios]


# ----------------------------------------------------------------------------- aggregation
def _rate(records, predicate) -> float:
    return sum(1 for r in records if predicate(r)) / len(records) if records else float("nan")


def summarize(records: list[dict]) -> dict:
    """Outcome rates, path efficiency, duration and defender separation for one group of episodes."""
    if not records:
        return {"n": 0}
    captured = [r for r in records if r["captured"]]
    excess = [r["path_excess"] for r in captured]
    return {
        "n": len(records),
        "capture_rate": _rate(records, lambda r: r["captured"]),
        "interception_rate": _rate(records, lambda r: r["outcome"] in ("intercepted", "unsafe_intercept")),
        "obstacle_collision_rate": _rate(records, lambda r: r["obstacle_collision"]),
        "endurance_exhaustion_rate": _rate(records, lambda r: r["endurance_exhausted"]),
        "timeout_rate": _rate(records, lambda r: r["outcome"] == "timeout"),
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "mean_path_excess_given_success": float(np.mean(excess)) if excess else float("nan"),
        "path_excess_flags_given_success": {f">{t}x": sum(e > t for e in excess) for t in PATH_EXCESS_FLAGS},
        "fraction_path_excess_above": {f">{t}x": (sum(e > t for e in excess) / len(excess)) if excess else float("nan")
                                       for t in PATH_EXCESS_FLAGS},
        "mean_flight_time_s": float(np.mean([r["flight_time_s"] for r in records])),
        "mean_min_defender_separation_m": float(np.mean([r["min_defender_separation_m"] for r in records])),
        "mean_return": float(np.mean([r["return_undiscounted"] for r in records])),
        "episodes_with_orbit_or_repeated_flyby": sum(1 for r in records if r["orbit_revolutions_max"] >= 1.0 or r["overshoots"] >= 3),
    }


def evaluate_acceptance(records: list[dict], a1_ids=("A1-201", "A1-202", "A1-203")) -> dict:
    """Apply the pre-registered gates. `records` must contain every A1 seed and matched A0 episodes."""
    t = A1_V2_ACCEPTANCE_THRESHOLDS
    by_attacker = defaultdict(list)
    for r in records:
        by_attacker[r["attacker"]].append(r)
    a0 = by_attacker["A0"]
    gates: dict = {}

    c0 = {a: [r for r in by_attacker[a] if r["level"] == "C0"] for a in a1_ids}
    gates["c0_directness_and_endurance"] = {
        "passed": all(
            summarize(c0[a])["endurance_exhaustion_rate"] <= t["c0_max_endurance_exhaustion_rate"]
            and summarize(c0[a])["fraction_path_excess_above"][">2.0x"] <= t["c0_max_fraction_path_excess_above_2x"]
            for a in a1_ids if c0[a]
        ) and all(c0[a] for a in a1_ids),
        "per_seed": {a: {k: summarize(c0[a]).get(k) for k in
                         ("endurance_exhaustion_rate", "fraction_path_excess_above", "episodes_with_orbit_or_repeated_flyby")}
                     for a in a1_ids},
        "a0_reference": {k: summarize([r for r in a0 if r["level"] == "C0"]).get(k)
                         for k in ("endurance_exhaustion_rate", "fraction_path_excess_above")},
    }

    advantage = {}
    for family in A1_V2_DEFENDER_FAMILIES:
        a0_capture = summarize([r for r in a0 if r["defender"] == family]).get("capture_rate", float("nan"))
        advantage[family] = {a: summarize([r for r in by_attacker[a] if r["defender"] == family]).get("capture_rate", float("nan"))
                             - a0_capture for a in a1_ids}
    gates["mission_advantage_over_a0"] = {
        "passed": any(sum(d >= t["mission_min_capture_advantage"] for d in per.values()) >= t["mission_min_seeds_with_advantage"]
                      for per in advantage.values()),
        "capture_advantage_by_defender": advantage,
    }

    pooled = {a: summarize(by_attacker[a]).get("capture_rate", float("nan")) for a in a1_ids}
    a0_pooled = summarize(a0).get("capture_rate", float("nan"))
    spread = (max(pooled.values()) - min(pooled.values())) if pooled else float("nan")
    gates["seed_generalization"] = {
        "passed": spread <= t["generalization_max_seed_spread"]
        and all(v >= a0_pooled - t["generalization_max_shortfall_vs_a0"] for v in pooled.values()),
        "pooled_capture_by_seed": pooled, "a0_pooled_capture": a0_pooled, "seed_spread": spread,
    }

    collision = {a: {lvl: summarize([r for r in by_attacker[a] if r["level"] == lvl]).get("obstacle_collision_rate", float("nan"))
                     for lvl in A1_V2_CLUTTER_LEVELS} for a in a1_ids}
    gates["obstacle_safety"] = {
        "passed": all(v <= t["obstacle_max_collision_rate"] for per in collision.values() for v in per.values()),
        "collision_rate_by_seed_and_level": collision,
        "a0_reference": {lvl: summarize([r for r in a0 if r["level"] == lvl]).get("obstacle_collision_rate")
                         for lvl in A1_V2_CLUTTER_LEVELS},
    }

    path = {a: summarize(by_attacker[a]).get("fraction_path_excess_above", {}) for a in a1_ids}
    gates["path_efficiency"] = {
        "passed": all(p.get(">2.0x", 0.0) <= t["path_max_fraction_above_2x_given_success"]
                      and p.get(">5.0x", 0.0) <= t["path_max_fraction_above_5x_given_success"] for p in path.values()),
        "fraction_above_given_success": path,
        "flag_counts_given_success": {a: summarize(by_attacker[a]).get("path_excess_flags_given_success") for a in a1_ids},
    }
    gates["all_passed"] = all(g["passed"] for g in gates.values() if isinstance(g, dict) and "passed" in g)
    gates["thresholds"] = dict(t)
    return gates


# ----------------------------------------------------------------------------- representative episodes
def select_representative_episodes(records: list[dict], a1_ids=("A1-201", "A1-202", "A1-203")) -> dict:
    """Deterministic choice of the episodes to inspect visually (A1 episodes only). Ties break on
    (attacker, level, defender, seed). Each entry carries the exact parameters to reload it."""
    a1 = [r for r in records if r["attacker"] in a1_ids]
    key = lambda r: (r["attacker"], r["level"], r["defender"], r["seed"])  # noqa: E731

    def pick(candidates, score):
        return min(candidates, key=lambda r: (score(r), key(r))) if candidates else None

    captured = [r for r in a1 if r["captured"]]
    selection = {
        "best_mission_success": pick(captured, lambda r: (-r["return_undiscounted"], r["path_excess"])),
        "successful_evasive": pick(
            [r for r in captured if r["min_defender_separation_m"] <= 15.0 and r["path_excess"] >= 1.05],
            lambda r: (r["min_defender_separation_m"], -r["path_excess"])),
        "endurance_exhaustion": pick([r for r in a1 if r["endurance_exhausted"]], lambda r: (-r["path_length_m"],)),
        "worst_path_excess": pick(a1, lambda r: (-(r["path_excess"] if math.isfinite(r["path_excess"]) else 0.0),)),
        "obstacle_heavy_success": pick([r for r in captured if r["level"] in ("C2", "C3")],
                                       lambda r: (-r["avoidance_steps"], r["path_excess"])),
    }
    return {role: (None if r is None else {k: r[k] for k in ("attacker", "level", "defender", "seed", "outcome",
                                                              "path_excess", "avoidance_steps", "min_defender_separation_m")})
            for role, r in selection.items()}


VISUAL_INSPECTION_MATRIX: tuple[tuple[str, str], ...] = (
    ("greedy", "C0"), ("greedy", "C2"), ("greedy", "C3"), ("lead", "C0"), ("lead", "C3"), ("rmpc", "C0"), ("rmpc", "C3"),
)  # each for A1-201, A1-202 and A1-203 (21 combinations; not all need to be watched)
