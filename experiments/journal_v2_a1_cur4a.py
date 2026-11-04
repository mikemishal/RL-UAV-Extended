"""Curriculum Stage 4A (`a1_curriculum_stage4a`) for the V2 learned attacker A1: pursue and capture the CANONICAL MOVING asset in C0 with NO active defender.

Why: Stage 4 (`journal_v2_a1_cur4.py`, moving asset + hold-until-detection Greedy) ended 0/3 mastered because the attackers repeatedly overshot the jittering asset (fly-bys at ~3.4 m miss distance, endurance
exhaustion 17-23 %) although defender-evasion behaviour was retained. Stage 4A isolates the moving-target pursuit that Stage 4 mixed with defender evasion; if it passes, Stage 4B recombines both.

Environment = the established Stage-1-v2 "curriculum_basic_pursuit_no_defender_interaction" mechanism (`journal_v2_a1_cur1v2.Stage1V2Env`: defender structurally present, parked, observation MASKED unavailable,
interception impossible via `intercept_radius = 1e-6`, perimeter spawn + initial heading offset U(0, 90 deg)) with exactly ONE change: the asset keeps its canonical V2 motion (`legacy_gaussian_random_walk`,
`v_s = 1.5 m/s`, sigma = 0.75 m per axis per step, reflecting boundaries, dedicated `_rng_soldier` stream) instead of `v_s = 0`. Kept: `a1_v2_obs_v1` (37-D), `a1_cur1_action_v1`, `a1_v2_reward_v2`, frozen PPO.

Initialisation: the accepted Stage3-v2 checkpoints (NOT the failed Stage-4 models). The Stage-4 modules and results are untouched and remain non-canonical evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur4 as P            # Stage 4 (failed, non-canonical evidence): parents, anchors, shared analysis helpers
from experiments import journal_v2_a1_r2 as R

VERSION = "a1_curriculum_stage4a"
CURRICULUM = "journal_v2_a1_curriculum"
STAGE = "stage4a"
ACTION_VERSION, OBSERVATION_VERSION, REWARD_VERSION = X.ACTION_VERSION, X.OBSERVATION_VERSION, X.REWARD_VERSION
TRAINING_PROTOCOL = "a1_cur4a_training_v1"
SCENARIO_PROTOCOL = "a1_cur4a_moving_asset_no_defender_interaction_v1"
CURRICULUM_MODE = "curriculum_moving_target_pursuit_no_defender_interaction"
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE
DOCS = "docs/journal_v2_a1_curriculum/stage4a/"
STAGE4_DOCS = P.DOCS
STAGE4_COMMIT = "15faece"
PARENT_STATUS, PARENT_SHA256, PARENT_MASTERY_ROLLOUT, STAGE3_COMMIT = P.PARENT_STATUS, P.PARENT_SHA256, P.PARENT_MASTERY_ROLLOUT, P.STAGE3_COMMIT
parent_checkpoint, parent_hashes = P.parent_checkpoint, P.parent_hashes
STAGE4_ACCEPTED_FINAL_SHA256 = {201: "e9f6f0c0c940101fee118ed8de83fcafaab9dc9e14025e706a0d57850f5d5a44", 202: "e768197ab908c39f16b92ac740a9ac57a1d1f868b40b89be5c246fb3cd550bb1",
                                203: "133302d45593c73040f4c821f10cb2c69f3b99e2a94facdb43cd1c1d77a5ffab"}   # failed Stage-4 final checkpoints (non-canonical evidence; never used for initialisation)

# ----------------------------------------------------------------------------- pre-registered constants (frozen before any Stage-4A training)
SCENARIO_STREAM_TAG = 0xC4A1
# The suggested 166000-166199 is NOT free (Stage-4 retention anchors 166000-166299), so the next clean contiguous block is used.
VALIDATION_A = (169_000, 169_099)
VALIDATION_B = (169_100, 169_199)
STAGE1_ANCHOR = (169_200, 169_299)           # Anchor 1: stationary basic pursuit (Stage-1-v2 task, evaluation only)
STATIC_ANCHOR = (169_300, 169_399)           # Anchor 2: Stage-2 stationary defender, stationary asset
MOVING_DEFENDER_ANCHOR = (169_400, 169_499)  # Anchor 3: Stage3-v2 hold/release Greedy, stationary asset
BASELINE = (171_000, 171_999)                # 1,000 matched pre-training scenarios
N_ENVS, MAX_ROLLOUTS, PPO_SEEDS = X.N_ENVS, 12, X.PPO_SEEDS
INTERCEPT_OUTCOMES = X.INTERCEPT_OUTCOMES
ANCHOR_MIN = {"anchor1_basic_pursuit": 0.98, "anchor2_static_defender": 0.80, "anchor3_moving_defender": 0.75}
ANCHOR_FAILURE_STREAK = 2
COMMON_PATHOLOGY_RATE, COMMON_PATHOLOGY_SEEDS = 0.01, 2

GATE = {
    "capture_min": 0.95, "exhaustion_max": 0.02, "timeout_max": 0.01, "median_path_excess_max": 1.50, "gt2_max": 0.05, "gt3_max": 0.02, "gt5_max": 0.0, "orbit_episodes_max": 0,
    "repeated_flyby_max": 0.05, "anchor1_capture_min": ANCHOR_MIN["anchor1_basic_pursuit"], "anchor2_capture_min": ANCHOR_MIN["anchor2_static_defender"],
    "anchor3_capture_min": ANCHOR_MIN["anchor3_moving_defender"], "nonfinite_max": 0,
}
RESERVED_RANGES = {**P.RESERVED_RANGES, "stage4_validation_A": P.VALIDATION_A, "stage4_validation_B": P.VALIDATION_B, "stage4_anchor1": P.STAGE1_ANCHOR, "stage4_anchor2": P.STATIC_ANCHOR,
                   "stage4_anchor3": P.MOVING_DEFENDER_ANCHOR, "stage4_diagnostic": P.DIAGNOSTIC, "stage4_scenario_validation": P.SCENARIO_VALIDATION}


def seeds_of(rng_tuple) -> range:
    return range(rng_tuple[0], rng_tuple[1] + 1)


def validation_seeds(name: str) -> range:
    return seeds_of({"A": VALIDATION_A, "B": VALIDATION_B}[name])


def final_evaluation_seeds() -> list[int]:
    return list(validation_seeds("A")) + list(validation_seeds("B"))


validation_set_for = X.validation_set_for
streak_failure_rollout = X.streak_failure_rollout
DEFENDER_SLOTS = [i for i, (name, _g) in enumerate(A.A1_V2_OBSERVATION_SCHEMA) if "defender" in name]


def namespace_overlaps() -> dict:
    own = {"validation_A": VALIDATION_A, "validation_B": VALIDATION_B, "anchor1": STAGE1_ANCHOR, "anchor2": STATIC_ANCHOR, "anchor3": MOVING_DEFENDER_ANCHOR, "baseline": BASELINE}
    out = {name: [k for k, (a, b) in RESERVED_RANGES.items() if not (hi < a or lo > b)] for name, (lo, hi) in own.items()}
    names = list(own)
    out["mutual"] = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if not (own[a][1] < own[b][0] or own[a][0] > own[b][1])]
    out["training_stream"] = [C.TRAIN_SEED_LOW, C.TRAIN_SEED_HIGH]
    out["training_overlap"] = [n for n, (lo, hi) in own.items() if not (hi < C.TRAIN_SEED_LOW or lo >= C.TRAIN_SEED_HIGH)]
    out["suggested_block_166000_166199"] = [k for k, (a, b) in RESERVED_RANGES.items() if not (166_199 < a or 166_000 > b)]
    return out


# ----------------------------------------------------------------------------- configuration
def stage4a_config():
    """Stage-1-v2 curriculum config (defender parked + masked + no interception) with the CANONICAL asset speed restored (`v_s` = canonical 1.5 m/s, `asset_motion_model` = legacy_gaussian_random_walk)."""
    canonical = A.build_episode_config("C0", "greedy")
    return dataclasses.replace(V1V2.stage1v2_config(), v_s=float(canonical.v_s))


def asset_motion_parameters() -> dict:
    cfg = stage4a_config()
    return {"model": cfg.asset_motion_model, "v_s_mps": float(cfg.v_s), "dt_s": float(cfg.dt), "sigma_m_per_axis_per_step": float(cfg.v_s) * float(cfg.dt), "axes": "horizontal (x, y); z = 0",
            "boundary": f"reflecting at [-{cfg.L}, {cfg.L}]^2", "start": "origin (0, 0, 0)", "rng_stream": "SeedSequence(seed).spawn(4)[1] (`_rng_soldier`), one normal(0, sigma, size=2) draw per step",
            "velocity_persistence": "none (i.i.d. displacements)", "obstacle_blocking": "inert in C0 (no obstacles)"}


def canonical_asset_motion_check() -> dict:
    canonical, cfg = A.build_episode_config("C0", "greedy"), stage4a_config()
    keys = ("v_s", "asset_motion_model", "asset_motion_tau_s", "dt", "L")
    return {k: {"canonical": getattr(canonical, k), "stage4a": getattr(cfg, k)} for k in keys} | {"identical": all(getattr(canonical, k) == getattr(cfg, k) for k in keys),
                                                                                                    "model_is_legacy_gaussian_random_walk": cfg.asset_motion_model == "legacy_gaussian_random_walk"}


def config_differences_from_canonical() -> dict:
    canonical, stage = A.build_episode_config("C0", "greedy"), stage4a_config()
    return {f.name: {"canonical": getattr(canonical, f.name), "stage4a": getattr(stage, f.name)} for f in dataclasses.fields(canonical) if getattr(canonical, f.name) != getattr(stage, f.name)}


def config_sha256() -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(stage4a_config()), sort_keys=True, default=str).encode()).hexdigest()


def scenario_stream(ppo_seed: int, worker_index: int, rollout_index: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SCENARIO_STREAM_TAG, int(ppo_seed), int(worker_index), int(rollout_index)]))


# ----------------------------------------------------------------------------- environment
def closest_approaches(ranges: np.ndarray) -> tuple[int, list[float]]:
    """(number of closest approaches, miss distances). An interior local minimum of the attacker-asset range is a fly-by pass (the attacker then recedes); the terminal minimum of an episode that ends while
    closing is the final approach. Interior minima above the 2 m capture radius are misses (a smaller one would have terminated the episode by swept capture)."""
    r = np.asarray(ranges, dtype=np.float64)
    interior = [float(r[i]) for i in range(1, len(r) - 1) if r[i] < r[i - 1] and r[i] <= r[i + 1]]
    final_closing = len(r) > 1 and r[-1] < r[-2]
    return len(interior) + int(final_closing), interior


class Stage4AEnv(V1V2.Stage1V2Env):
    """Stage-1-v2 clean curriculum environment (masked parked defender, no interception) with the canonical MOVING asset."""

    def __init__(self, ppo_seed: int, worker_index: int = 0, rollout_index: int = 0, scripted_a0: bool = False, record_trajectory: bool = False, gamma: float = R.GAMMA):
        super().__init__(ppo_seed, worker_index, rollout_index, scripted_a0, record_trajectory, gamma)
        self._config = stage4a_config()
        self._stream = scenario_stream(ppo_seed, worker_index, rollout_index)

    def reseed(self, rollout_index: int) -> int:
        self._stream = scenario_stream(self.ppo_seed, self.worker_index, rollout_index)
        return int(rollout_index)

    def asset_trace(self) -> np.ndarray:
        """Asset positions at every step boundary (initial state + after each step), float64."""
        return np.vstack([np.asarray(self._asset0, dtype=np.float64)[None, :]] + [np.asarray(p, dtype=np.float64)[None, :] for p in self._ast])

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        out = super()._episode_summary(outcome, path_now, range_next)
        trace = self.asset_trace()
        ranges = np.array([self._initial_range] + self._ranges)
        count, misses = closest_approaches(ranges)
        out.update(asset_final_position=[float(x) for x in trace[-1]], asset_path_length_m=float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum()),
                   asset_final_displacement_m=float(np.linalg.norm(trace[-1] - trace[0])), asset_moving=True, closest_approach_count=int(count), miss_distances_m=misses,
                   min_range_m=float(ranges.min()))
        return out


def make_env(ppo_seed: int = 0, worker_index: int = 0, rollout_index: int = 0, **kwargs) -> Stage4AEnv:
    return Stage4AEnv(ppo_seed, worker_index, rollout_index, **kwargs)


# ----------------------------------------------------------------------------- evaluation
def evaluate(act, seeds, *, scripted_a0: bool = False, record_trajectory: bool = False) -> list[dict]:
    env = Stage4AEnv(0, 0, 0, scripted_a0=scripted_a0, record_trajectory=record_trajectory)
    return [C.run_episode(env, s, None if scripted_a0 else act) for s in seeds]


def evaluate_anchor1(act, seeds=None) -> list[dict]:
    return V1V2.evaluate(act, list(seeds if seeds is not None else seeds_of(STAGE1_ANCHOR)))


def evaluate_anchor2(act, seeds=None) -> list[dict]:
    return S.evaluate(act, list(seeds if seeds is not None else seeds_of(STATIC_ANCHOR)))


def evaluate_anchor3(act, seeds=None) -> list[dict]:
    return X.evaluate(act, list(seeds if seeds is not None else seeds_of(MOVING_DEFENDER_ANCHOR)))


anchor_metrics = P.metrics      # metrics of the defender anchors (Stage-2 static / Stage3-v2 hold-release), exactly as in Stage 4


def metrics(episodes: list[dict]) -> dict:
    out = C.metrics(episodes)
    n = len(episodes)
    pe = np.array([e["path_excess"] for e in episodes], dtype=np.float64)
    cap = np.array([e["outcome"] == "soldier_caught" for e in episodes])
    first_pass = np.array([e["overshoots"] == 0 for e in episodes])
    out["gt3"] = float(np.mean(pe > 3.0))
    out["gt3_count"] = int((pe > 3.0).sum())
    out["repeated_flyby_fraction"] = out["repeated_flyby_episodes"] / n
    out["flyby_fraction"] = out["flyby_episodes"] / n
    out["interception"] = out["intercepted"]
    out["first_pass_capture"] = float(np.mean(cap & first_pass))
    out["first_pass_fraction_of_captures"] = float(np.mean(first_pass[cap])) if cap.any() else None
    out["closest_approaches_mean"] = float(np.mean([e["closest_approach_count"] for e in episodes]))
    misses = [m for e in episodes for m in e["miss_distances_m"]]
    out["miss_distance_median_m"] = float(np.median(misses)) if misses else None
    out["miss_distance_count"] = len(misses)
    out["asset_displacement_max_mean_m"] = float(np.mean([e["asset_displacement_max_m"] for e in episodes]))
    out["asset_final_displacement_mean_m"] = float(np.mean([e.get("asset_final_displacement_m", 0.0) for e in episodes]))
    return out


def gate_status(m: dict, *, a0_capture: float, anchor1: float, anchor2: float, anchor3: float) -> dict:
    """`a0_capture` is recorded for context only; there is no improvement-over-A0 requirement in a no-defender pursuit task."""
    checks = {
        "capture": m["capture"] >= GATE["capture_min"], "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"], "timeout": m["timeout"] <= GATE["timeout_max"],
        "median_path_excess": m["median_path_excess"] <= GATE["median_path_excess_max"], "gt2": m["gt2"] <= GATE["gt2_max"], "gt3": m["gt3"] <= GATE["gt3_max"], "gt5": m["gt5"] <= GATE["gt5_max"],
        "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"], "repeated_flyby": m["repeated_flyby_fraction"] <= GATE["repeated_flyby_max"],
        "anchor1_basic_pursuit": anchor1 >= GATE["anchor1_capture_min"], "anchor2_static_defender": anchor2 >= GATE["anchor2_capture_min"],
        "anchor3_moving_defender": anchor3 >= GATE["anchor3_capture_min"], "numerical_failures": m["nonfinite"] <= GATE["nonfinite_max"],
    }
    return {"checks": checks, "a0_capture_context": a0_capture, "passed": bool(all(checks.values()))}


# ----------------------------------------------------------------------------- common random numbers
def asset_traces(seeds, policies: dict, env_class=None) -> dict:
    out = {}
    for name, act in policies.items():
        env = (env_class or Stage4AEnv)(0, 0, 0, scripted_a0=act is None)
        out[name] = {}
        for s in seeds:
            C.run_episode(env, int(s), act)
            out[name][int(s)] = env.asset_trace()
    return out


crn_report = P.crn_report


def masking_check(seeds, policy_seed: int = 0) -> dict:
    """Largest absolute value of any defender observation slot over every decision of the given scenarios (must be exactly 0), and the number of interception outcomes."""
    env = Stage4AEnv(0, 0, 0)
    rng = np.random.default_rng(policy_seed)
    worst, steps, intercepts = 0.0, 0, 0
    for s in seeds:
        obs, _ = env.reset(options={"scenario": int(s)})
        done = False
        while not done:
            worst = max(worst, float(np.abs(obs[DEFENDER_SLOTS]).max()))
            obs, _r, te, tr, info = env.step(rng.uniform(-1, 1, 3).astype(np.float32))
            steps += 1
            done = te or tr
        intercepts += int(info["outcome"] in INTERCEPT_OUTCOMES)
    return {"scenarios": len(list(seeds)), "decisions": steps, "max_abs_defender_observation": worst, "interception_outcomes": intercepts, "defender_slots": DEFENDER_SLOTS}


# ----------------------------------------------------------------------------- pre-training baseline
def baseline_summary(episodes: list[dict]) -> dict:
    n = len(episodes)
    outcome = np.array([e["outcome"] for e in episodes])
    m = metrics(episodes)
    pe = np.array([e["path_excess"] for e in episodes], dtype=np.float64)
    return {"scenarios": n, "capture": m["capture"], "interception": float(np.isin(outcome, INTERCEPT_OUTCOMES).mean()), "timeout": m["timeout"], "endurance_exhaustion": m["exhaustion"],
            "outcomes": {k: int((outcome == k).sum()) for k in sorted(set(outcome.tolist()))}, "median_path_excess": m["median_path_excess"], "path_excess_p90": float(np.nanpercentile(pe, 90)),
            "gt2": m["gt2"], "gt3": m["gt3"], "gt5": m["gt5"], "flyby_episodes": m["flyby_episodes"], "repeated_flyby_episodes": m["repeated_flyby_episodes"], "repeated_flyby_fraction": m["repeated_flyby_fraction"],
            "orbit_episodes": m["orbit_episodes"], "closest_approaches_mean": m["closest_approaches_mean"], "miss_distance_median_m": m["miss_distance_median_m"], "miss_distance_count": m["miss_distance_count"],
            "first_pass_capture": m["first_pass_capture"], "first_pass_fraction_of_captures": m["first_pass_fraction_of_captures"], "path_length_m": P._dist([e["path_length_m"] for e in episodes]),
            "duration_s": P._dist([e["duration_s"] for e in episodes]), "asset_displacement_max_m": P._dist([e["asset_displacement_max_m"] for e in episodes]),
            "asset_final_displacement_m": P._dist([e.get("asset_final_displacement_m") for e in episodes]), "nonfinite_episodes": int(sum(not e["finite"] for e in episodes))}


def baseline_validation(seeds=None) -> dict:
    from stable_baselines3 import PPO

    seeds = list(seeds if seeds is not None else seeds_of(BASELINE))
    a0_eps = evaluate(None, seeds, scripted_a0=True)
    out = {"scenarios": len(seeds), "a0": baseline_summary(a0_eps), "parents": {}}
    actors = {}
    for s in PPO_SEEDS:
        model = PPO.load(parent_checkpoint(s)["path"].removesuffix(".zip"), device="cpu")
        actors[s] = C.model_actor(model)
        out["parents"][str(s)] = baseline_summary(evaluate(actors[s], seeds)) | {"parent_sha256": PARENT_SHA256[s]}
    sub = seeds[:50]
    again = evaluate(None, sub, scripted_a0=True)
    out["deterministic_replay_first_50"] = all(a["outcome"] == b["outcome"] and a["steps"] == b["steps"] and a["asset_final_position"] == b["asset_final_position"] for a, b in zip(a0_eps[:50], again))
    policies = {"a0": None} | {f"parent_{s}": actors[s] for s in PPO_SEEDS}
    out["crn"] = crn_report(asset_traces(sub, policies))
    out["crn_vs_stage4_environment"] = crn_report({"stage4a_a0": asset_traces(sub, {"a0": None})["a0"], "stage4_a0": asset_traces(sub, {"a0": None}, env_class=P.Stage4Env)["a0"]})
    out["masking"] = masking_check(seeds[:30])
    return out


def baseline_verdict(summary: dict) -> dict:
    """Configuration validity only - the stage never modifies the environment because of the difficulty; the baseline just documents the missing competence."""
    a0, parents = summary["a0"], summary["parents"]
    checks = {
        "asset_moves": a0["asset_displacement_max_m"]["median"] > 0.0, "canonical_asset_motion": bool(canonical_asset_motion_check()["identical"]),
        "defender_masked_in_every_state": summary["masking"]["max_abs_defender_observation"] == 0.0, "interception_impossible": summary["masking"]["interception_outcomes"] == 0
        and a0["interception"] == 0.0 and all(p["interception"] == 0.0 for p in parents.values()),
        "no_nonfinite_state": a0["nonfinite_episodes"] == 0 and all(p["nonfinite_episodes"] == 0 for p in parents.values()),
        "crn_asset_trajectories_identical": bool(summary["crn"]["identical"]), "crn_matches_stage4_asset_realisation": bool(summary["crn_vs_stage4_environment"]["identical"]),
        "deterministic_replay": bool(summary["deterministic_replay_first_50"]),
    }
    return {"checks": checks, "passed": bool(all(checks.values())), "a0_capture": a0["capture"], "parent_capture": {k: p["capture"] for k, p in parents.items()},
            "competence_gap_vs_gate": {k: round(GATE["capture_min"] - p["capture"], 4) for k, p in parents.items()},
            "rule": "configuration validity only; difficulty is reported, never used to change the environment"}


# ----------------------------------------------------------------------------- protocol
def protocol_specification(parents: dict | None = None) -> dict:
    parents = parents if parents is not None else parent_hashes()
    v3 = X.protocol_specification()
    return {
        "version": VERSION,
        "hypothesis": "The Stage-4 failure is primarily caused by insufficient moving-target pursuit competence rather than loss of defender-evasion skill. Isolating canonical moving-target pursuit should allow the "
                      "curriculum policy to acquire tracking competence before the moving defender is reintroduced.",
        "tests": "curriculum decomposition of Stage 4: moving-target pursuit only (no active defender)",
        "identifiers": {"training_protocol": TRAINING_PROTOCOL, "scenario_protocol": SCENARIO_PROTOCOL, "action": ACTION_VERSION, "observation": OBSERVATION_VERSION, "reward": REWARD_VERSION,
                        "curriculum": CURRICULUM, "stage": STAGE, "mode": CURRICULUM_MODE},
        "decomposition": {"failed_stage4": {"commit": STAGE4_COMMIT, "result": "0/3 mastered; held-out capture 52.5 / 60.5 / 59.5 %; repeated fly-bys at ~3.4 m miss distance, endurance exhaustion 17-23 %",
                                            "status": "non-canonical evidence; checkpoints never used here", "final_sha256": STAGE4_ACCEPTED_FINAL_SHA256},
                          "next_stage_if_pass": "Stage 4B = moving asset + Stage3-v2 hold-until-detection Greedy + C0 (not started here)"},
        "parents": {"status": PARENT_STATUS, "checkpoints": parents, "note": "accepted Stage3-v2 curriculum checkpoints (NOT the failed Stage-4 models); remain curriculum models, not final A1"},
        "environment": {
            "clutter": "C0 (no static or dynamic obstacles)", "asset": "canonical V2 motion", "asset_motion": asset_motion_parameters(), "asset_motion_check_vs_canonical": canonical_asset_motion_check(),
            "config_differences_from_canonical": config_differences_from_canonical(), "env_config_sha256": config_sha256(),
            "defender": "structurally present, stationary, parked at the Stage-1 ground corner (-48,-48,0), never acts; observation masked unavailable (defender flag/position/velocity slots exactly 0); "
                        "interception impossible (intercept_radius = 1e-6 m) - the established Stage-1-v2 mechanism, reused unchanged",
            "geometry": "Stage-1-v2: canonical perimeter spawn, altitude U(10, 30), initial heading offset U(0, 90) deg", "scenario_stream": f"SeedSequence([{hex(SCENARIO_STREAM_TAG)}, ppo_seed, worker, rollout])",
            "common_random_numbers": "same evaluation seed => same initial attacker state, same asset state and the same asset random walk (dedicated `_rng_soldier` stream) for A0, the Stage3-v2 parent and the Stage-4A "
                                     "policy; asserted by tests and by the baseline CRN report (also against the Stage-4 environment)"},
        "observation_audit": "a1_v2_obs_v1 (37-D) unchanged: it carries the CURRENT asset position; the walk is memoryless so no future/asset-velocity feature is needed and none is exposed.",
        "initialization": {"actor_critic_log_std": "policy state_dict of the matching accepted Stage3-v2 checkpoint (actor, CRITIC, log_std)", "optimizer": "RESET (fresh Adam)", "timestep_counter": "reset to 0",
                           "not_loaded": ["failed Stage-4 models", "BC checkpoints", "Stage-2 models", "Stage-3 optimizer state"]},
        "reward": REWARD_VERSION + " unchanged (no moving-target shaping, no prediction/velocity-matching term, no A0 fallback)",
        "ppo": {**v3["ppo"], "torch_threads": 2},
        "training_budget": {"max_rollouts": MAX_ROLLOUTS, "steps_per_rollout": 32_768, "max_timesteps": MAX_ROLLOUTS * 32_768},
        "validation": {"A": list(VALIDATION_A), "B": list(VALIDATION_B), "alternation": "rollout k: odd -> A, even -> B", "anchor1_basic_pursuit": list(STAGE1_ANCHOR), "anchor2_static_defender": list(STATIC_ANCHOR),
                       "anchor3_moving_defender": list(MOVING_DEFENDER_ANCHOR), "baseline": list(BASELINE), "suggested_block_conflict": "166000-166199 is occupied by the Stage-4 retention anchors; the next clean contiguous blocks are used",
                       "final": "169000-169199 (200 episodes) + matched canonical A0 + matched Stage3-v2 parent (common random numbers)", "namespace_overlaps": namespace_overlaps()},
        "baseline_rule": "configuration validity only (asset moves, canonical motion, defender masked, no interception, CRN, determinism); difficulty is documented and never used to modify the environment",
        "gate": GATE, "mastery": "two consecutive rollout validations pass ALL gate checks",
        "retention": {"anchors": ANCHOR_MIN, "stop": f"capture below the anchor minimum on {ANCHOR_FAILURE_STREAK} consecutive rollouts on ANY anchor => CURRICULUM_RETENTION_FAILURE"},
        "success_rule": ">= 2/3 seeds master, moving-target tracking improves (mastered seeds: paired Stage4A - parent capture CI lower bound > 0 AND repeated fly-by rate below the parent's), anchors retained, "
                        "no shared pathology (same category in > 1 % of episodes for >= 2 seeds); < 2 mastered => STOP, do not reintroduce the defender",
        "verdict_B_rule": "not passed AND (>= 1 seed mastered OR >= 2 seeds with held-out capture >= 0.90 and paired CI lower bound > 0 vs the parent)",
        "paired_statistics": v3["paired_statistics"],
    }


def protocol_sha256(parents: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(protocol_specification(parents), sort_keys=True, default=str).encode()).hexdigest()
