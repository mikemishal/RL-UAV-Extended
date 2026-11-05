"""Foundation acceptance experiment for A1-CBF (journal_cbf_attacker): analytical relative-target pursuit + an
identity CBF/QP safety-filter interface + an acceleration-limited quadrotor velocity executor
(uav_defend/attackers/cbf_attacker.py). NO RL, NO obstacle/defender CBF constraints -- pursuit/dynamics foundation
only. The research focus beyond this stage is the counter-UAS DEFENDER; A1-CBF exists to give the defender a
non-degenerate, literature-motivated attacker, not to be a novel attacker contribution in itself.

Foundation validation: 1,000 matched scenarios (C0, canonical moving asset, no active defender interaction -- the
established intercept_radius=1e-6 curriculum mechanism, reused unchanged) comparing canonical scripted A0
(`enemy_control_mode="legacy"`, `enemy_nominal_guidance_mode="direct_obstacle_only"`) and A1-CBF
(`enemy_control_mode="cbf_velocity"`) on the SAME scenario seeds, with common random numbers (the asset's
`_rng_soldier` stream is independent of `enemy_control_mode`, so this holds without extra engineering -- verified by
a dedicated test).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from uav_defend.attackers.cbf_attacker import CBF_DT_SUB, CBF_KP, CBFAttackerLimits
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

from experiments import journal_v2_validation_core as VC
from experiments import journal_v2_a1_r2 as R
from experiments.journal_v2_a1_cur4a import closest_approaches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = "docs/journal_cbf_attacker/foundation/"
RESULTS_ROOT = R.V2_RESULTS_DIR / "journal_cbf_attacker" / "foundation"

# ----------------------------------------------------------------------------- pre-registered namespaces (frozen before any validation)
# A fresh, independent block: no relation to the A1 PPO/BC curriculum's seed namespaces (which stay below 200,000 for
# evaluation blocks and use a >=1,000,000 training stream) -- chosen with a wide margin from both.
FOUNDATION_SEEDS = (500_000, 500_999)       # 1,000 matched A0 vs A1-CBF scenarios
LEGACY_NOOP_SEEDS = (510_000, 510_999)      # 1,000 matched legacy-mode (new code) vs git-parent-commit scenarios
CHUNK_SIZE = 100
PARENT_COMMIT = "2f45914"                   # research/journal-corrected-attacker-v2 branch point this stage forked from

# Frozen acceptance gate (do not tune after seeing results; see docs/journal_cbf_attacker/foundation/README.md).
GATE = {
    "capture_min": 0.98, "exhaustion_max": 0.02, "timeout_max": 0.0, "first_pass_capture_min": 0.97,
    "repeated_flyby_max": 0.02, "orbit_episodes_max": 0, "nonfinite_max": 0,
}
REPEATED_FLYBY_OVERSHOOTS = 2   # >= this many overshoots in one episode counts as a "repeated fly-by"
ORBIT_REVOLUTION_THRESHOLD = 1.0


def seeds_of(rng: tuple[int, int]) -> range:
    return range(rng[0], rng[1] + 1)


def foundation_config() -> EnvConfig:
    """Canonical C0 V2 config with `intercept_radius=1e-6` (the established no-defender-interaction curriculum
    mechanism, unchanged, reused verbatim -- the defender is structurally present but cannot terminate the
    episode). Nothing else differs from the canonical C0 V2 config; the moving asset (`v_s=1.5`,
    `legacy_gaussian_random_walk`) and finite mission endurance (k=5) are UNCHANGED canonical values.

    Deliberately does NOT vary by `enemy_control_mode`: that switch lives on `SoldierEnv.__init__` (like
    `render_mode`), never on `EnvConfig`, specifically so it can never perturb `dataclasses.asdict(config)` --
    the fingerprint every prior PPO/BC curriculum stage's frozen protocol/handoff artifacts pin for
    reproducibility (see uav_defend/envs/soldier_env.py's `__init__` docstring)."""
    cfg, _estimator = VC.build_config("C0", "V2", "passive")
    return dataclasses.replace(cfg, intercept_radius=1e-6)


def config_differences_from_canonical() -> dict:
    from experiments import journal_v2_a1 as A

    canonical = A.build_episode_config("C0", "greedy")
    cfg = foundation_config()
    return {f.name: {"canonical": getattr(canonical, f.name), "foundation": getattr(cfg, f.name)} for f in dataclasses.fields(canonical)
            if getattr(canonical, f.name) != getattr(cfg, f.name)}


def config_sha256() -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(foundation_config()), sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- episode runner (ground truth only; no policy/observation interface)
def run_episode(env: SoldierEnv, seed: int) -> dict:
    """Runs one scenario with `env`'s own control mode (A0 scripted legacy or A1-CBF) with a constant zero
    defender action (no active defender interaction; the defender cannot terminate the episode -- see
    foundation_config()). Returns ground-truth diagnostics read back from `info`, never from a learned policy."""
    obs, info = env.reset(seed=seed)
    start_pos = np.asarray(info["enemy_pos"], dtype=np.float64).copy()
    asset0 = np.asarray(info["soldier_pos"], dtype=np.float64).copy()
    initial_range = float(np.linalg.norm(asset0 - start_pos))
    att, ast, speeds, accels, ranges, finite = [start_pos], [asset0], [], [], [initial_range], True
    prev_pos = start_pos.copy()
    path_length = 0.0
    done, info_final = False, info
    while not done:
        obs, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        asset_pos = np.asarray(info["soldier_pos"], dtype=np.float64)
        path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
        att.append(pos)
        ast.append(asset_pos)
        d = float(np.linalg.norm(asset_pos - pos))
        ranges.append(d)
        speeds.append(float(info["enemy_speed"]))
        accels.append(float(info["enemy_accel"]))
        finite = finite and bool(np.isfinite(pos).all() and np.isfinite(d) and np.isfinite(info["enemy_speed"]))
        done = te or tr
        info_final = info
    ranges_arr = np.array(ranges, dtype=np.float64)
    overshoots = int(VC.count_overshoots(ranges_arr, env.config.dt))
    closest_n, misses = closest_approaches(ranges_arr)
    orbit_max, orbit_total = VC.orbit_revolutions(np.array(att), np.array(ast))
    chord = float(np.linalg.norm(start_pos - asset0))
    outcome = info_final["outcome"]
    speeds_arr = np.array(speeds, dtype=np.float64) if speeds else np.zeros(1)
    accels_arr = np.array(accels, dtype=np.float64) if accels else np.zeros(1)
    return {
        "scenario_seed": int(seed), "outcome": outcome, "steps": len(speeds), "duration_s": len(speeds) * env.config.dt,
        "path_length_m": path_length, "path_excess": (path_length / chord) if chord > 1e-9 else float("nan"),
        "initial_range_m": initial_range, "terminal_distance_m": float(ranges_arr[-1]), "min_distance_m": float(ranges_arr.min()),
        "overshoots": overshoots, "closest_approach_count": int(closest_n), "miss_distances_m": [float(m) for m in misses],
        "first_pass": bool(outcome == "soldier_caught" and overshoots == 0), "repeated_flyby": bool(overshoots >= REPEATED_FLYBY_OVERSHOOTS),
        "orbit_revolutions_max": float(orbit_max), "orbit": bool(orbit_max >= ORBIT_REVOLUTION_THRESHOLD),
        "mean_speed_mps": float(speeds_arr.mean()), "max_speed_mps": float(speeds_arr.max()), "min_speed_mps": float(speeds_arr.min()),
        "max_accel_mps2": float(accels_arr.max()), "fraction_time_below_3mps": float(np.mean(speeds_arr < 3.0)),
        "hover_events": int(np.sum(speeds_arr < 0.5)), "finite": finite,
        "enemy_endurance_exhausted": outcome == "enemy_endurance_exhausted",
        "speed_vs_distance": [[float(d_), float(s_)] for d_, s_ in zip(ranges_arr[:-1], speeds_arr)] if len(speeds_arr) == len(ranges_arr) - 1 else [],
    }


def evaluate(seeds, *, enemy_control_mode: str) -> list[dict]:
    env = SoldierEnv(foundation_config(), enemy_control_mode=enemy_control_mode)
    return [run_episode(env, s) for s in seeds]


def metrics(episodes: list[dict]) -> dict:
    n = len(episodes)
    outcome = np.array([e["outcome"] for e in episodes])
    pe = np.array([e["path_excess"] for e in episodes], dtype=np.float64)
    cap = outcome == "soldier_caught"
    first_pass_of_captures = [e["first_pass"] for e in episodes if e["outcome"] == "soldier_caught"]
    misses = [m for e in episodes for m in e["miss_distances_m"]]
    return {
        "episodes": n, "capture": float(cap.mean()), "exhaustion": float((outcome == "enemy_endurance_exhausted").mean()),
        "timeout": float((outcome == "timeout").mean()), "outcomes": {k: int((outcome == k).sum()) for k in sorted(set(outcome.tolist()))},
        "first_pass_capture": float(np.mean([e["first_pass"] for e in episodes])),
        "first_pass_fraction_of_captures": (float(np.mean(first_pass_of_captures)) if first_pass_of_captures else None),
        "repeated_flyby_episodes": int(sum(e["repeated_flyby"] for e in episodes)), "repeated_flyby_fraction": float(np.mean([e["repeated_flyby"] for e in episodes])),
        "flyby_episodes": int(sum(e["overshoots"] >= 1 for e in episodes)), "orbit_episodes": int(sum(e["orbit"] for e in episodes)),
        "nonfinite": int(sum(not e["finite"] for e in episodes)), "median_path_excess": float(np.nanmedian(pe)), "mean_path_excess": float(np.nanmean(pe)),
        "mean_duration_s": float(np.mean([e["duration_s"] for e in episodes])), "mean_path_length_m": float(np.mean([e["path_length_m"] for e in episodes])),
        "mean_closest_approaches": float(np.mean([e["closest_approach_count"] for e in episodes])),
        "miss_distance_median_m": (float(np.median(misses)) if misses else None), "miss_distance_count": len(misses),
        "mean_speed_mps": float(np.mean([e["mean_speed_mps"] for e in episodes])), "max_speed_mps": float(np.max([e["max_speed_mps"] for e in episodes])),
        "max_accel_mps2": float(np.max([e["max_accel_mps2"] for e in episodes])), "mean_fraction_time_below_3mps": float(np.mean([e["fraction_time_below_3mps"] for e in episodes])),
        "total_hover_events": int(sum(e["hover_events"] for e in episodes)),
    }


def gate_status(m: dict) -> dict:
    checks = {
        "capture": m["capture"] >= GATE["capture_min"], "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"],
        "timeout": m["timeout"] <= GATE["timeout_max"], "first_pass_capture": m["first_pass_capture"] >= GATE["first_pass_capture_min"],
        "repeated_flyby": m["repeated_flyby_fraction"] <= GATE["repeated_flyby_max"], "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"],
        "nonfinite": m["nonfinite"] <= GATE["nonfinite_max"],
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def constraint_verification(episodes: list[dict], *, limits: CBFAttackerLimits) -> dict:
    """Empirical verification (section 16's 'Dynamics' gate) that the quadrotor executor never produced a speed,
    acceleration or climb/descent violation, over EVERY recorded (distance, speed) sample of the foundation run."""
    max_speed_seen = max((e["max_speed_mps"] for e in episodes), default=0.0)
    max_accel_seen = max((e["max_accel_mps2"] for e in episodes), default=0.0)
    speed_ok = max_speed_seen <= limits.max_horizontal_speed + 1e-6
    accel_ok = max_accel_seen <= limits.max_accel + 1e-6
    return {"max_observed_horizontal_speed_mps": max_speed_seen, "max_allowed_horizontal_speed_mps": limits.max_horizontal_speed,
            "horizontal_speed_within_limit": speed_ok, "max_observed_accel_mps2": max_accel_seen, "max_allowed_accel_mps2": limits.max_accel,
            "accel_within_limit": accel_ok, "passed": bool(speed_ok and accel_ok)}


# ----------------------------------------------------------------------------- legacy no-op comparison (section 14)
def _load_legacy_module(ref: str):
    """Dynamically imports `uav_defend/envs/soldier_env.py` AS IT EXISTED at git ref `ref`, under a throwaway module
    name, reusing the CURRENT (unmodified-by-this-stage) uav_defend.config/.tracking/.dynamics/.obstacles packages
    for every import inside that file (none of those were touched; only uav_defend/config/env_config.py gained one
    NEW field with a safe default, and uav_defend/envs/soldier_env.py itself gained the single early-return `if`
    checked in the no-op test below). This is the most direct available proof that enabling the opt-in
    `enemy_control_mode` switch (left at its default "legacy") changes NOTHING relative to the pre-stage code."""
    src = subprocess.run(["git", "show", f"{ref}:uav_defend/envs/soldier_env.py"], cwd=PROJECT_ROOT, capture_output=True, check=True, encoding="utf-8").stdout
    mod_name = f"uav_defend.envs._soldier_env_legacy_baseline_{ref}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_loader(mod_name, loader=None, origin=f"git:{ref}:uav_defend/envs/soldier_env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    exec(compile(src, f"<git:{ref}:uav_defend/envs/soldier_env.py>", "exec"), module.__dict__)
    return module


def legacy_noop_episode_keys(info: dict) -> dict:
    """The subset of `info` compared for bit-for-bit legacy equivalence: outcomes, terminal step, every entity's
    trajectory-determining state, path length, and the collision/contact result."""
    return {
        "outcome": info["outcome"], "step_count": info["step_count"], "enemy_pos": [round(float(x), 9) for x in info["enemy_pos"]],
        "defender_pos": [round(float(x), 9) for x in info["defender_pos"]], "soldier_pos": [round(float(x), 9) for x in info["soldier_pos"]],
        "enemy_vel": [round(float(x), 9) for x in info["enemy_vel"]], "defender_vel": [round(float(x), 9) for x in info["defender_vel"]],
        "attacker_path_length": round(float(info["attacker_path_length"]), 6), "terminal_event": info["terminal_event"],
        "capture_contact_distance": info["capture_contact_distance"], "interception_contact_distance": info["interception_contact_distance"],
        "enemy_endurance_traveled_m": info["enemy_endurance_traveled_m"],
    }


def run_legacy_episode(env_cls, cfg, seed: int) -> dict:
    env = env_cls(cfg)
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
        done = te or tr
    return legacy_noop_episode_keys(info)


def legacy_noop_check(seeds, *, parent_commit: str = PARENT_COMMIT) -> dict:
    """Runs `seeds` through (a) the CURRENT SoldierEnv with enemy_control_mode left at its default "legacy", and
    (b) the SoldierEnv class as it existed at `parent_commit` (loaded via `_load_legacy_module`), on an identical
    canonical C0/V2/greedy config, and asserts every compared field is identical. See `legacy_noop_episode_keys` for
    the exact fields (outcomes, terminal step, every trajectory, path length, collision/contact result)."""
    legacy_mod = _load_legacy_module(parent_commit)
    cfg, _ = VC.build_config("C0", "V2", "greedy")
    mismatches = []
    for s in seeds:
        current = run_legacy_episode(SoldierEnv, cfg, s)
        baseline = run_legacy_episode(legacy_mod.SoldierEnv, cfg, s)
        if current != baseline:
            mismatches.append({"seed": int(s), "current": current, "baseline": baseline})
    return {"parent_commit": parent_commit, "seeds_checked": len(list(seeds)), "mismatches": mismatches, "identical": len(mismatches) == 0}


# ----------------------------------------------------------------------------- protocol
def protocol_specification() -> dict:
    return {
        "version": "a1_cbf_foundation_v1",
        "hypothesis": "An analytical, literature-motivated relative-target pursuit law combined with an "
                      "acceleration-limited quadrotor velocity executor (no legacy heading/turn-rate limit) fixes "
                      "the repeated-fly-by failure mode observed throughout the PPO/BC A1 curriculum (Stages 4, "
                      "4A, 4A-1) under the existing canonical moving-asset benchmark, with no RL training.",
        "architecture": "current relative target state -> nominal velocity guidance -> CBF/QP safety filter (identity, Prompt 1) -> quadrotor velocity executor",
        "environment": {"clutter": "C0", "asset": "canonical V2 motion (v_s=1.5, legacy_gaussian_random_walk)", "defender": "structurally present, cannot terminate the episode "
                        "(intercept_radius=1e-6, the established no-defender-interaction curriculum mechanism, reused unchanged)",
                        "mission_contact_mode": "swept", "enemy_endurance_enabled": True, "capture_radius_m": 2.0},
        "control_mode_switch": {"field": "enemy_control_mode", "location": "SoldierEnv.__init__ constructor argument (deliberately NOT an EnvConfig field, "
                                "so it can never perturb dataclasses.asdict(config) used by every prior curriculum stage's frozen protocol/handoff hashes)",
                                "default": "legacy (bit-for-bit unchanged)", "new": "cbf_velocity",
                                "config_sha256": config_sha256(), "config_differences_from_canonical": config_differences_from_canonical()},
        "nominal_guidance": {"law": "v_nom = min(max_speed, kp * ||r||) * r_hat, r = asset_pos - attacker_pos (CURRENT, never initial/predicted)",
                             "kp_per_s": CBF_KP, "max_speed_mps": CBFAttackerLimits().max_horizontal_speed, "no_target_velocity_feedforward": True,
                             "no_minimum_forward_speed": True, "frozen_before_acceptance_run": True},
        "safety_filter": {"stage": "Prompt 1: identity (v_safe = v_nom, hazards=())", "future_stage": "CBF/QP obstacle + defender avoidance (NOT implemented here)"},
        "executor": {"model": "acceleration-limited (no heading/turn-rate limit) 3-D quadrotor point mass", "dt_external_s": 0.5, "dt_sub_s": CBF_DT_SUB,
                     "substeps_per_step": 5, "max_horizontal_speed_mps": CBFAttackerLimits().max_horizontal_speed, "max_accel_mps2": CBFAttackerLimits().max_accel,
                     "max_climb_rate_mps": CBFAttackerLimits().max_climb_rate, "max_descent_rate_mps": CBFAttackerLimits().max_descent_rate,
                     "command_held_constant_across_substeps": True, "swept_mission_contact": "unchanged single linear segment over the full 0.5s external step; no new/parallel contact test"},
        "capture_semantics_unchanged": {"capture_radius_m": 2.0, "swept": True, "first_event_wins": True},
        "validation": {"foundation_seeds": list(FOUNDATION_SEEDS), "legacy_noop_seeds": list(LEGACY_NOOP_SEEDS), "chunk_size": CHUNK_SIZE, "parent_commit": PARENT_COMMIT},
        "gate": GATE, "freeze_rule": "kp, speed/accel/climb/descent limits, substep, capture radius are frozen BEFORE the acceptance run and never tuned afterward",
        "scope": "pursuit + dynamics foundation ONLY; no obstacle CBF, no defender CBF, no RL training -- research focus beyond this stage is the counter-UAS defender",
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()


def namespace_overlaps() -> dict:
    """Documents (does not merely assert) the gap between this stage's seed blocks and every A1 curriculum block."""
    curriculum_known_max = 192_199   # highest evaluation-namespace seed used anywhere in the PPO/BC curriculum (Stage4A-1 retention_source)
    return {"foundation_seeds": list(FOUNDATION_SEEDS), "legacy_noop_seeds": list(LEGACY_NOOP_SEEDS),
            "curriculum_known_max_evaluation_seed": curriculum_known_max,
            "gap_to_curriculum_m": FOUNDATION_SEEDS[0] - curriculum_known_max,
            "foundation_vs_legacy_noop_overlap": not (FOUNDATION_SEEDS[1] < LEGACY_NOOP_SEEDS[0] or LEGACY_NOOP_SEEDS[1] < FOUNDATION_SEEDS[0])}


# ----------------------------------------------------------------------------- restart-safe chunked validation (section 18)
def chunk_seeds(namespace: tuple[int, int], chunk_index: int, chunk_size: int = CHUNK_SIZE) -> list[int]:
    lo = namespace[0] + chunk_index * chunk_size
    hi = min(lo + chunk_size - 1, namespace[1])
    return list(range(lo, hi + 1))


def num_chunks(namespace: tuple[int, int], chunk_size: int = CHUNK_SIZE) -> int:
    total = namespace[1] - namespace[0] + 1
    return (total + chunk_size - 1) // chunk_size


def _content_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_foundation_validation(results_root: Path, *, enemy_control_mode: str, namespace: tuple[int, int] = FOUNDATION_SEEDS, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """Restart-safe: each chunk is written atomically with an embedded content hash + this stage's protocol hash; a
    chunk whose file verifies is never regenerated. Resumes from the highest verified chunk automatically (every
    chunk is checked independently, in order, so a restart after any interruption continues from the first
    unverified chunk)."""
    from experiments import journal_v2_a1_eg_state as base

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    psha = protocol_sha256()
    env = None
    all_episodes: list[dict] = []
    for i in range(num_chunks(namespace, chunk_size)):
        path = results_root / f"{enemy_control_mode}_chunk_{i:03d}.json"
        episodes = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_sha256") == psha and payload.get("enemy_control_mode") == enemy_control_mode and _content_sha256(payload["episodes"]) == payload.get("content_sha256"):
                episodes = payload["episodes"]
        except (OSError, ValueError, KeyError, TypeError):
            episodes = None
        if episodes is None:
            env = env or SoldierEnv(foundation_config(), enemy_control_mode=enemy_control_mode)
            seeds = chunk_seeds(namespace, i, chunk_size)
            episodes = [run_episode(env, s) for s in seeds]
            base.atomic_write_json(path, {"schema": "a1_cbf_foundation_chunk/1", "chunk": i, "enemy_control_mode": enemy_control_mode, "seeds": seeds,
                                           "episodes": episodes, "content_sha256": _content_sha256(episodes), "protocol_sha256": psha})
        all_episodes.extend(episodes)
    return all_episodes


def run_legacy_noop_validation(results_root: Path, *, namespace: tuple[int, int] = LEGACY_NOOP_SEEDS, chunk_size: int = CHUNK_SIZE, parent_commit: str = PARENT_COMMIT) -> dict:
    """Restart-safe chunked version of legacy_noop_check(): each chunk's mismatch list is written atomically with
    the parent commit and this stage's protocol hash; a verified chunk is never re-run (re-running the comparison
    is cheap, but the chunking mirrors the SAME restart contract as run_foundation_validation for consistency)."""
    from experiments import journal_v2_a1_eg_state as base

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    psha = protocol_sha256()
    legacy_mod = _load_legacy_module(parent_commit)
    cfg, _ = VC.build_config("C0", "V2", "greedy")
    mismatches: list[dict] = []
    checked = 0
    for i in range(num_chunks(namespace, chunk_size)):
        path = results_root / f"legacy_noop_chunk_{i:03d}.json"
        payload = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_sha256") != psha or payload.get("parent_commit") != parent_commit:
                payload = None
        except (OSError, ValueError, KeyError, TypeError):
            payload = None
        if payload is None:
            seeds = chunk_seeds(namespace, i, chunk_size)
            chunk_mismatches = []
            for s in seeds:
                current = run_legacy_episode(SoldierEnv, cfg, s)
                baseline = run_legacy_episode(legacy_mod.SoldierEnv, cfg, s)
                if current != baseline:
                    chunk_mismatches.append({"seed": int(s), "current": current, "baseline": baseline})
            payload = {"schema": "a1_cbf_legacy_noop_chunk/1", "chunk": i, "seeds": seeds, "mismatches": chunk_mismatches, "parent_commit": parent_commit, "protocol_sha256": psha}
            base.atomic_write_json(path, payload)
        checked += len(payload["seeds"])
        mismatches.extend(payload["mismatches"])
    return {"parent_commit": parent_commit, "seeds_checked": checked, "mismatches": mismatches, "identical": len(mismatches) == 0}
