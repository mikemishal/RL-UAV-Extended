"""Curriculum Stage 4 (`a1_curriculum_stage4`) for the V2 learned attacker A1: the protected asset MOVES with the canonical V2 asset-motion model, everything else as Stage3-v2.

Exactly ONE new source of difficulty relative to Stage3-v2 (`journal_v2_a1_cur3v2.py`): the stationary asset (`v_s = 0`) is replaced by the canonical V2 asset motion. Kept: C0 (no obstacles), the HOLD-UNTIL-DETECTION Greedy
defender (curriculum scaffold, NOT removed here), canonical attacker dynamics, `a1_v2_obs_v1` (37-D), `a1_cur1_action_v1`, `a1_v2_reward_v2`, frozen PPO.

Canonical asset motion (live code `SoldierEnv._move_soldier_legacy`, config `asset_motion_model = "legacy_gaussian_random_walk"`): every step the asset draws an i.i.d. horizontal Gaussian displacement with standard
deviation sigma = v_s * dt = 1.5 m/s * 0.5 s = 0.75 m per axis (x, y), z stays 0, reflecting boundaries at [-L, L]^2 (L = 50 m), start at the origin. The walk has NO velocity persistence, so the current asset position is the
complete Markov state of the asset (the 37-D observation already contains it; no asset-velocity feature carries information). Draws come from the dedicated `_rng_soldier` stream (SeedSequence(seed).spawn(4)[1]), which is
independent of the attacker, defender and sensor streams -> common random numbers across policies on the same evaluation seed.

Defender placement is the unchanged Stage3-v2/Stage-2 procedure: nominal-A0 path from the INITIAL state with a stationary asset (`nominal_a0_path`), lambda ~ U(0.35, 0.65), r ~ U(0, 1.5 m). It never sees future asset motion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_validation_core as VC

VERSION = "a1_curriculum_stage4"
CURRICULUM = "journal_v2_a1_curriculum"
STAGE = "stage4"
ACTION_VERSION, OBSERVATION_VERSION, REWARD_VERSION = X.ACTION_VERSION, X.OBSERVATION_VERSION, X.REWARD_VERSION
TRAINING_PROTOCOL = "a1_cur4_training_v1"
SCENARIO_PROTOCOL = "a1_cur4_moving_asset_hold_until_detection_greedy_v1"
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE
DOCS = "docs/journal_v2_a1_curriculum/stage4/"
STAGE3_DOCS = X.DOCS
STAGE3_COMMIT = "3a2b23f"
PARENT_STATUS = "quantitatively_accepted_curriculum_parent"
PARENT_SHA256 = {
    201: "c60ae85d0c0b84c2b52dd4d0c280ccede9f6ecae6166448bd3a9de8218988e78",
    202: "3dfabf5cf245bc07aa02ac67840792c9a422724a7784be2efe07d196389ba485",
    203: "818338a5579e1e63259dff8fb94a61f2518a2e61db9901e158703a177951b52c",
}
PARENT_MASTERY_ROLLOUT = {201: 3, 202: 5, 203: 7}

# ----------------------------------------------------------------------------- pre-registered constants (frozen before any Stage-4 training)
SCENARIO_STREAM_TAG = 0xC4B1
VALIDATION_A = (165_000, 165_099)
VALIDATION_B = (165_100, 165_199)
STAGE1_ANCHOR = (166_000, 166_099)          # Anchor 1: Stage-1 no-defender, stationary asset (evaluation only)
STATIC_ANCHOR = (166_100, 166_199)          # Anchor 2: Stage-2 stationary defender, stationary asset
MOVING_DEFENDER_ANCHOR = (166_200, 166_299)  # Anchor 3: Stage3-v2 hold/release Greedy, stationary asset
DIAGNOSTIC = (167_000, 167_099)
SCENARIO_VALIDATION = (168_000, 168_999)
N_ENVS, MAX_ROLLOUTS, PPO_SEEDS = X.N_ENVS, 16, X.PPO_SEEDS
DIAGNOSTIC_CONTROLLERS = X.DIAGNOSTIC_CONTROLLERS
INTERCEPT_OUTCOMES = X.INTERCEPT_OUTCOMES
ANCHOR_MIN = {"anchor1_basic_pursuit": 0.98, "anchor2_static_defender": 0.80, "anchor3_moving_defender": 0.75}
ANCHOR_FAILURE_STREAK = 2
A0_DIFFICULTY = {"a0_capture_max": 0.50, "a0_interception_min": 0.50, "rule": "A0 capture <= 0.50 OR A0 interception >= 0.50"}
PARENT_MIN_CAPTURE = 0.20                    # at least one Stage3-v2 parent must capture > 20 % on the Stage-4 task

GATE = {
    "capture_min": 0.70, "improvement_over_a0_min_pp": 20.0, "interception_max": 0.30, "timeout_max": 0.03, "exhaustion_max": 0.03, "median_path_excess_max": 2.00, "gt3_max": 0.05, "gt5_max": 0.0,
    "orbit_episodes_max": 0, "anchor1_capture_min": ANCHOR_MIN["anchor1_basic_pursuit"], "anchor2_capture_min": ANCHOR_MIN["anchor2_static_defender"],
    "anchor3_capture_min": ANCHOR_MIN["anchor3_moving_defender"], "nonfinite_max": 0,
}
RESERVED_RANGES = {**X.RESERVED_RANGES, "stage3_v2_validation_A": X.VALIDATION_A, "stage3_v2_validation_B": X.VALIDATION_B, "stage3_v2_stage1_anchor": X.STAGE1_ANCHOR,
                   "stage3_v2_static_anchor": X.STATIC_ANCHOR, "stage3_v2_diagnostic": X.DIAGNOSTIC, "stage3_v2_scenario_validation": X.SCENARIO_VALIDATION}


def seeds_of(rng_tuple) -> range:
    return range(rng_tuple[0], rng_tuple[1] + 1)


def validation_seeds(name: str) -> range:
    return seeds_of({"A": VALIDATION_A, "B": VALIDATION_B}[name])


def final_evaluation_seeds() -> list[int]:
    return list(validation_seeds("A")) + list(validation_seeds("B"))


validation_set_for = X.validation_set_for
streak_failure_rollout = X.streak_failure_rollout


def namespace_overlaps() -> dict:
    own = {"validation_A": VALIDATION_A, "validation_B": VALIDATION_B, "anchor1": STAGE1_ANCHOR, "anchor2": STATIC_ANCHOR, "anchor3": MOVING_DEFENDER_ANCHOR, "diagnostic": DIAGNOSTIC,
           "scenario_validation": SCENARIO_VALIDATION}
    out = {name: [k for k, (a, b) in RESERVED_RANGES.items() if not (hi < a or lo > b)] for name, (lo, hi) in own.items()}
    names = list(own)
    out["mutual"] = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if not (own[a][1] < own[b][0] or own[a][0] > own[b][1])]
    out["training_stream"] = [C.TRAIN_SEED_LOW, C.TRAIN_SEED_HIGH]
    out["training_overlap"] = [n for n, (lo, hi) in own.items() if not (hi < C.TRAIN_SEED_LOW or lo >= C.TRAIN_SEED_HIGH)]
    return out


# ----------------------------------------------------------------------------- configuration
def stage4_config(controller: str = "greedy"):
    """Canonical C0 V2 config of the controller's estimator track with ONLY the standby synchronisation disabled (as Stage3-v2). `v_s`, `asset_motion_model` and every asset-motion parameter stay canonical."""
    cfg, _estimator = VC.build_config("C0", "V2", controller)
    return dataclasses.replace(cfg, defender_standby_until_detection=False)


def asset_motion_parameters() -> dict:
    cfg = stage4_config()
    sigma = float(cfg.v_s) * float(cfg.dt)
    return {"model": cfg.asset_motion_model, "v_s_mps": float(cfg.v_s), "dt_s": float(cfg.dt), "sigma_m_per_axis_per_step": sigma, "axes": "horizontal (x, y); z = 0",
            "boundary": f"reflecting at [-{cfg.L}, {cfg.L}]^2", "start": "origin (0, 0, 0)", "rng_stream": "SeedSequence(seed).spawn(4)[1] (`_rng_soldier`), one normal(0, sigma, size=2) draw per step",
            "velocity_persistence": "none (i.i.d. displacements; asset_motion_tau_s = %s is used only by gauss_markov_velocity)" % cfg.asset_motion_tau_s,
            "obstacle_blocking": "inert in C0 (no obstacles)"}


def canonical_asset_motion_check() -> dict:
    canonical = A.build_episode_config("C0", "greedy")
    cfg = stage4_config()
    keys = ("v_s", "asset_motion_model", "asset_motion_tau_s", "dt", "L")
    return {k: {"canonical": getattr(canonical, k), "stage4": getattr(cfg, k)} for k in keys} | {"identical": all(getattr(canonical, k) == getattr(cfg, k) for k in keys),
                                                                                                      "model_is_legacy_gaussian_random_walk": cfg.asset_motion_model == "legacy_gaussian_random_walk"}


def config_differences_from_canonical(controller: str = "greedy") -> dict:
    canonical = A.build_episode_config("C0", "greedy" if controller == "greedy" else controller)
    stage = stage4_config(controller)
    return {f.name: {"canonical": getattr(canonical, f.name), "stage4": getattr(stage, f.name)} for f in dataclasses.fields(canonical) if getattr(canonical, f.name) != getattr(stage, f.name)}


def config_sha256(controller: str = "greedy") -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(stage4_config(controller)), sort_keys=True, default=str).encode()).hexdigest()


def scenario_stream(ppo_seed: int, worker_index: int, rollout_index: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SCENARIO_STREAM_TAG, int(ppo_seed), int(worker_index), int(rollout_index)]))


# ----------------------------------------------------------------------------- environment
class Stage4Env(X.Stage3V2Env):
    """Stage3-v2 environment (hold-until-detection scripted defender on the nominal-A0-path placement) with the canonical MOVING asset."""

    def __init__(self, ppo_seed: int, worker_index: int = 0, rollout_index: int = 0, scripted_a0: bool = False, record_trajectory: bool = False, gamma: float = R.GAMMA, *,
                 controller: str = "greedy", hold_until_detection: bool = True):
        super().__init__(ppo_seed, worker_index, rollout_index, scripted_a0, record_trajectory, gamma, controller=controller, hold_until_detection=hold_until_detection)
        self._config = stage4_config(controller)
        self._stream = scenario_stream(ppo_seed, worker_index, rollout_index)

    def reseed(self, rollout_index: int) -> int:
        self._stream = scenario_stream(self.ppo_seed, self.worker_index, rollout_index)
        return int(rollout_index)

    def asset_trace(self) -> np.ndarray:
        """Asset positions at every step boundary (initial state + after each step), float64."""
        return np.vstack([np.asarray(self._asset0, dtype=np.float64)[None, :]] + [np.asarray(p, dtype=np.float64)[None, :] for p in self._ast])

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        out = super()._episode_summary(outcome, path_now, range_next)
        trace = np.vstack([np.asarray(self._asset0, dtype=np.float64)[None, :]] + [np.asarray(p, dtype=np.float64)[None, :] for p in self._ast])
        steps_len = np.linalg.norm(np.diff(trace, axis=0), axis=1)
        out.update(asset_final_position=[float(x) for x in trace[-1]], asset_path_length_m=float(steps_len.sum()), asset_final_displacement_m=float(np.linalg.norm(trace[-1] - trace[0])),
                   asset_moving=True)
        return out


def make_env(ppo_seed: int = 0, worker_index: int = 0, rollout_index: int = 0, **kwargs) -> Stage4Env:
    return Stage4Env(ppo_seed, worker_index, rollout_index, **kwargs)


class Stage4NoDefenderEnv(V1V2.Stage1V2Env):
    """Counterfactual: the SAME moving asset (same `_rng_soldier` stream, hence the same realized trajectory) with the defender masked and interception disabled - used only to analyse what a policy does WITHOUT the defender."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = dataclasses.replace(V1V2.stage1v2_config(), v_s=float(stage4_config().v_s))


# ----------------------------------------------------------------------------- evaluation
def evaluate(act, seeds, *, scripted_a0: bool = False, record_trajectory: bool = False, controller: str = "greedy", hold_until_detection: bool = True) -> list[dict]:
    env = Stage4Env(0, 0, 0, scripted_a0=scripted_a0, record_trajectory=record_trajectory, controller=controller, hold_until_detection=hold_until_detection)
    return [C.run_episode(env, s, None if scripted_a0 else act) for s in seeds]


def evaluate_anchor1(act, seeds=None) -> list[dict]:
    return V1V2.evaluate(act, list(seeds if seeds is not None else seeds_of(STAGE1_ANCHOR)))


def evaluate_anchor2(act, seeds=None) -> list[dict]:
    return S.evaluate(act, list(seeds if seeds is not None else seeds_of(STATIC_ANCHOR)))


def evaluate_anchor3(act, seeds=None) -> list[dict]:
    return X.evaluate(act, list(seeds if seeds is not None else seeds_of(MOVING_DEFENDER_ANCHOR)))


def metrics(episodes: list[dict]) -> dict:
    out = X.metrics(episodes)
    out["asset_displacement_max_mean_m"] = float(np.mean([e["asset_displacement_max_m"] for e in episodes]))
    out["asset_final_displacement_mean_m"] = float(np.mean([e.get("asset_final_displacement_m", 0.0) for e in episodes]))
    return out


def gate_status(m: dict, *, a0_capture: float, anchor1: float, anchor2: float, anchor3: float) -> dict:
    improvement_pp = round(100.0 * (m["capture"] - a0_capture), 9)
    checks = {
        "capture": m["capture"] >= GATE["capture_min"], "improvement_over_a0": improvement_pp >= GATE["improvement_over_a0_min_pp"], "interception": m["interception"] <= GATE["interception_max"],
        "timeout": m["timeout"] <= GATE["timeout_max"], "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"], "median_path_excess": m["median_path_excess"] <= GATE["median_path_excess_max"],
        "gt3": m["gt3"] <= GATE["gt3_max"], "gt5": m["gt5"] <= GATE["gt5_max"], "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"],
        "anchor1_basic_pursuit": anchor1 >= GATE["anchor1_capture_min"], "anchor2_static_defender": anchor2 >= GATE["anchor2_capture_min"],
        "anchor3_moving_defender": anchor3 >= GATE["anchor3_capture_min"], "numerical_failures": m["nonfinite"] <= GATE["nonfinite_max"],
    }
    return {"checks": checks, "improvement_over_a0_pp": improvement_pp, "passed": bool(all(checks.values()))}


# ----------------------------------------------------------------------------- common random numbers
def asset_traces(seeds, policies: dict) -> dict:
    """Asset trajectories realised under each named policy (`None` = canonical A0) on identical evaluation seeds."""
    out = {}
    for name, act in policies.items():
        env = Stage4Env(0, 0, 0, scripted_a0=act is None)
        out[name] = {}
        for s in seeds:
            C.run_episode(env, int(s), act)
            out[name][int(s)] = env.asset_trace()
    return out


def crn_report(traces: dict, placements: dict | None = None) -> dict:
    """Every policy sees exactly the same realized asset trajectory on all COMPLETED steps of the common prefix of each seed. Episodes end at different times and the final element of a trace is the swept-contact
    event-time state of the terminal step (interpolated by the simulator, so it legitimately differs from the full-step position another policy reaches); it is therefore excluded from the exact comparison and its
    deviation is reported separately (bounded by one step of motion)."""
    names = list(traces)
    seeds = list(traces[names[0]])
    worst, compared, terminal_worst = 0.0, 0, 0.0
    for s in seeds:
        for n in names[1:]:
            a, b = traces[names[0]][s], traces[n][s]
            k = min(len(a), len(b))
            if k > 1:
                worst = max(worst, float(np.abs(a[:k - 1] - b[:k - 1]).max()))
                compared += k - 1
            terminal_worst = max(terminal_worst, float(np.abs(a[k - 1] - b[k - 1]).max()))
    return {"policies": names, "seeds": len(seeds), "compared_asset_positions": compared, "max_abs_difference_m": worst, "identical": worst == 0.0,
            "terminal_event_state_max_difference_m": terminal_worst, "terminal_bound_m": float(asset_motion_parameters()["sigma_m_per_axis_per_step"]) * 6.0}


# ----------------------------------------------------------------------------- scenario validation (pre-training)
def _dist(values):
    return X._dist(values)


def baseline_summary(episodes: list[dict]) -> dict:
    n = len(episodes)
    outcome = np.array([e["outcome"] for e in episodes])
    cap, icpt = outcome == "soldier_caught", np.isin(outcome, INTERCEPT_OUTCOMES)
    return {"scenarios": n, "capture": float(cap.mean()), "interception": float(icpt.mean()), "timeout": float((outcome == "timeout").mean()), "endurance_exhaustion": float((outcome == "enemy_endurance_exhausted").mean()),
            "outcomes": {k: int((outcome == k).sum()) for k in sorted(set(outcome.tolist()))}, "path_length_m": _dist([e["path_length_m"] for e in episodes]),
            "asset_displacement_max_m": _dist([e["asset_displacement_max_m"] for e in episodes]), "asset_final_displacement_m": _dist([e.get("asset_final_displacement_m") for e in episodes]),
            "min_defender_separation_m": _dist([e["min_defender_separation_m"] for e in episodes]), "duration_s": _dist([e["duration_s"] for e in episodes]), "median_path_excess": float(np.median([e["path_excess"] for e in episodes])),
            "nonfinite_episodes": int(sum(not e["finite"] for e in episodes))}


def scenario_validation(seeds=None) -> dict:
    from stable_baselines3 import PPO

    seeds = list(seeds if seeds is not None else seeds_of(SCENARIO_VALIDATION))
    a0_eps = evaluate(None, seeds, scripted_a0=True)
    dt = float(stage4_config().dt)
    x_summary = X.summarize_scenarios(a0_eps, dt)
    out = {"scenarios": len(seeds), "a0": baseline_summary(a0_eps), "a0_hold_release_validity": x_summary["validity"], "a0_detection_time_s": x_summary["detection_time_s"],
           "a0_heading_bin_breakdown": x_summary["heading_bin_breakdown"], "parents": {}}
    for s in PPO_SEEDS:
        model = PPO.load(parent_checkpoint(s)["path"].removesuffix(".zip"), device="cpu")
        eps = evaluate(C.model_actor(model), seeds)
        out["parents"][str(s)] = baseline_summary(eps) | {"parent_sha256": PARENT_SHA256[s]}
    sub = seeds[:50]
    again = evaluate(None, sub, scripted_a0=True)
    first = a0_eps[:len(sub)]
    out["deterministic_replay_first_50"] = all(a["outcome"] == b["outcome"] and a["steps"] == b["steps"] and a["asset_final_position"] == b["asset_final_position"] for a, b in zip(first, again))
    return out


def validation_verdict(summary: dict) -> dict:
    a0 = summary["a0"]
    v = summary["a0_hold_release_validity"]
    parent_caps = {k: p["capture"] for k, p in summary["parents"].items()}
    checks = {
        "a0_difficulty": a0["capture"] <= A0_DIFFICULTY["a0_capture_max"] or a0["interception"] >= A0_DIFFICULTY["a0_interception_min"],
        "a_parent_captures_gt_20pct": max(parent_caps.values()) > PARENT_MIN_CAPTURE, "no_nonfinite_state": a0["nonfinite_episodes"] == 0 and all(p["nonfinite_episodes"] == 0 for p in summary["parents"].values()),
        "no_immediate_contact_degeneracy": v["episodes_steps_le_1"] == 0 and v["intercepted_with_fewer_than_2_visible_decisions"] == 0,
        "defender_velocity_at_release_zero": v["max_defender_speed_at_release_mps"] == 0.0, "defender_fixed_before_release": v["max_defender_displacement_before_release_m"] == 0.0,
        "greedy_takes_over_after_release": v["released_episodes_where_greedy_moved_defender"] == v["released_episodes_with_time_after_release"],
        "asset_moves": a0["asset_displacement_max_m"]["median"] > 0.0, "crn_asset_trajectories_identical": bool(summary.get("crn", {}).get("identical", False)),
        "deterministic_replay": bool(summary["deterministic_replay_first_50"]),
    }
    return {"checks": checks, "passed": bool(all(checks.values())), "a0_dominates": bool(not checks["a0_difficulty"]), "parents_collapse": bool(not checks["a_parent_captures_gt_20pct"]),
            "parent_capture": parent_caps, "rule": {"a0": A0_DIFFICULTY, "parent": f"at least one Stage3-v2 parent captures > {PARENT_MIN_CAPTURE}"}}


# ----------------------------------------------------------------------------- Stage3-v2 parents
def parent_checkpoint(seed: int, project_root: Path | None = None) -> dict:
    """The accepted Stage3-v2 checkpoint of `seed`: file SHA-256 verified against the pre-registered full hash, the Stage3-v2 seed manifest and the committed docs manifest."""
    from experiments import journal_v2_a1_cur1_training as T1

    root = Path(project_root) if project_root is not None else T1.PROJECT_ROOT
    results = (root / "results" / "journal_v2_corrected_semantics" / "a1_curriculum" / "stage3_v2") if project_root is not None else X.RESULTS_ROOT
    manifest = json.loads((results / f"seed_{seed}" / "checkpoints" / "seed_manifest.json").read_text())
    path = results / f"seed_{seed}" / "checkpoints" / "accepted_model.zip"
    sha = T1.T.sha256_file(path)
    committed = json.loads((root / "docs" / "journal_v2_a1_curriculum" / "stage3_v2" / f"seed_{seed}_manifest.json").read_text())
    ok = (sha == PARENT_SHA256[int(seed)] == manifest["accepted_model_sha256"] == committed["accepted_model_sha256"] and manifest["status"] == "MASTERED"
          and manifest["accepted_rollout"] == PARENT_MASTERY_ROLLOUT[int(seed)] and manifest["training_protocol"] == X.TRAINING_PROTOCOL)
    if not ok:
        raise RuntimeError(f"Stage3-v2 parent of seed {seed} failed verification (file {sha})")
    return {"seed": int(seed), "attacker_id": f"A1-CUR3V2-{seed}", "path": str(path), "model_sha256": sha, "rollout": manifest["accepted_rollout"], "stage3_v2_training_status": manifest["status"],
            "parent_status": PARENT_STATUS, "stage3_v2_commit": STAGE3_COMMIT, "stage3_v2_protocol_sha256": manifest["protocol_sha256"]}


def parent_hashes() -> dict:
    return {str(s): {"model_sha256": parent_checkpoint(s)["model_sha256"], "rollout": PARENT_MASTERY_ROLLOUT[s], "status": PARENT_STATUS} for s in PPO_SEEDS}


# ----------------------------------------------------------------------------- protocol
def protocol_specification(parents: dict | None = None) -> dict:
    parents = parents if parents is not None else parent_hashes()
    v3 = X.protocol_specification()
    return {
        "version": VERSION,
        "hypothesis": "After learning pursuit and moving-defender evasion in simplified conditions, curriculum initialization enables PPO to adapt to stochastic target motion without catastrophic loss of previously learned "
                      "pursuit/evasion behavior.",
        "tests": "target-motion complexity only",
        "identifiers": {"training_protocol": TRAINING_PROTOCOL, "scenario_protocol": SCENARIO_PROTOCOL, "action": ACTION_VERSION, "observation": OBSERVATION_VERSION, "reward": REWARD_VERSION,
                        "curriculum": CURRICULUM, "stage": STAGE},
        "single_conceptual_change_from_stage3_v2": "The protected asset moves with the canonical V2 asset-motion model (legacy_gaussian_random_walk, v_s = 1.5 m/s) instead of being stationary.",
        "stage3_v2": {"commit": STAGE3_COMMIT, "protocol_sha256": X.protocol_sha256(), "status": "quantitatively accepted curriculum parent (3/3 mastered; manual-review acceptance not recorded)",
                      "outputs_never_mixed": "separate results / scratch / docs namespaces; verify_rollout rejects any manifest with a different protocol sha"},
        "parents": {"status": PARENT_STATUS, "checkpoints": parents,
                    "note": "Stage3-v2 checkpoints remain curriculum models, not final A1; earlier mastery results are not rewritten"},
        "environment": {
            "clutter": "C0 (no static or dynamic obstacles)", "asset": "canonical V2 motion", "asset_motion": asset_motion_parameters(), "asset_motion_check_vs_canonical": canonical_asset_motion_check(),
            "config": "canonical C0 V2 measurement-track config with ONLY defender_standby_until_detection=False (as Stage3-v2)", "env_config_sha256": config_sha256(),
            "differences_from_canonical": config_differences_from_canonical(), "detection_radius_m": X.detection_radius(),
            "defender": v3["environment"]["defender"], "placement": "unchanged Stage3-v2 / Stage-2 nominal-A0-path placement computed from the INITIAL state with a stationary-asset nominal path; "
                                                                     "no dependence on future asset motion (no future-information leakage)",
            "common_random_numbers": "same evaluation seed => same initial attacker state, asset state, defender placement and asset-motion stream (`_rng_soldier`, spawned from the episode seed and independent of attacker/defender/sensor "
                                     "streams); asserted by tests and by the pre-training CRN report",
            "curriculum_scaffold": "hold-until-detection is retained from Stage3-v2 and is NOT removed here"},
        "observation_audit": "a1_v2_obs_v1 (37-D) is kept. It contains the CURRENT asset position (and own state, gated defender, endurance). The canonical asset walk has no velocity persistence (i.i.d. Gaussian displacements), "
                             "so the current position is the complete Markov state: an asset-velocity feature would carry no predictive information and no future information is exposed. No augmentation needed.",
        "initialization": {"actor_critic_log_std": "policy state_dict of the matching Stage3-v2 accepted checkpoint (actor, CRITIC, log_std); the critic is reused because observation and reward semantics are unchanged "
                                                   "(its value estimates are stale for moving-asset states and adapt during PPO)",
                           "optimizer": "RESET (fresh Adam) at the curriculum boundary", "timestep_counter": "reset to 0 (Stage-4 rollouts 1..16)", "not_loaded": ["BC checkpoints", "Stage-2 models", "Stage-3 optimizer state"]},
        "reward": REWARD_VERSION + " unchanged (no moving-target shaping, no prediction/velocity-matching/defender-distance term)",
        "ppo": {**v3["ppo"], "torch_threads": 2},
        "validation": {"A": list(VALIDATION_A), "B": list(VALIDATION_B), "alternation": "rollout k: odd -> A, even -> B", "anchor1_basic_pursuit": list(STAGE1_ANCHOR), "anchor2_static_defender": list(STATIC_ANCHOR),
                       "anchor3_moving_defender": list(MOVING_DEFENDER_ANCHOR), "diagnostic": list(DIAGNOSTIC), "scenario_validation": list(SCENARIO_VALIDATION),
                       "final": "165000-165199 (200 episodes) + matched canonical A0 + matched Stage3-v2 parent (common random numbers)", "namespace_overlaps": namespace_overlaps()},
        "scenario_validation_rule": {"a0": A0_DIFFICULTY, "parents": f"at least one Stage3-v2 parent captures > {PARENT_MIN_CAPTURE}", "validity": "hold/release semantics, no degeneracy, asset moves, CRN identical, deterministic",
                                     "on_failure": "STOP before training and report; asset motion and placement are never modified"},
        "gate": GATE, "mastery": "two consecutive rollout validations pass ALL gate checks",
        "retention": {"anchors": ANCHOR_MIN, "stop": f"capture below the anchor minimum on {ANCHOR_FAILURE_STREAK} consecutive rollouts on ANY anchor => CURRICULUM_RETENTION_FAILURE"},
        "generalization_diagnostic": {"controllers": list(DIAGNOSTIC_CONTROLLERS), "seeds": list(DIAGNOSTIC), "semantics": "moving canonical asset, same hold-until-detection scaffold; diagnostic only"},
        "success_rule": ">= 2/3 seeds master, mastered seeds retain all three anchors, no shared structural pathology; < 2 => STOP, no clutter",
        "paired_statistics": v3["paired_statistics"],
    }


def protocol_sha256(parents: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(protocol_specification(parents), sort_keys=True, default=str).encode()).hexdigest()
