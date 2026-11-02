"""Curriculum Stage 3 for the V2 learned attacker A1: evade a MOVING scripted (canonical Greedy) defender while capturing a STATIONARY asset in open space (C0, no obstacles).

Identifiers: action `a1_cur1_action_v1`, observation `a1_v2_obs_v1`, reward `a1_v2_reward_v2` (all unchanged), training protocol `a1_cur3_training_v1`, scenario protocol
`a1_cur3_moving_greedy_nominal_path_v1`. See docs/journal_v2_a1_curriculum/stage3/. Configuration / wrapper only: `uav_defend` and the Greedy controller are untouched.

  * asset    : stationary (`v_s = 0`), canonical origin.
  * defender : the canonical scripted `greedy` controller (`uav_defend.policies.registry.get_policy("greedy")`) driving the canonical constrained defender dynamics (v_d = 18 m/s, 8 m/s^2, 90 deg/s) with
               canonical swept interception (2.5 m). Greedy escorts the asset until the attacker is detected (defender-side detection radius 15 m) and then pursues the measured attacker position.
  * placement: identical to Stage2-v2 (`journal_v2_a1_cur2.place_defender`): lam ~ U(0.35, 0.65) of the arc length of A0's noise-free nominal path, perpendicular offset r ~ U(0, 1.5 m). The defender STARTS AT REST
               there. `defender_standby_until_detection = False` (as in Stage 1/2): with the canonical standby the defender would be re-synchronised onto the asset on the first step, erasing the placement. The Greedy
               controller then acts from the first step ("legacy" action semantics of the same environment switch), i.e. it runs toward the asset in escort mode until it detects the attacker.
  * obs      : `build_a1_v2_observation(env, 25 m)` (true-range gated defender position AND velocity features - now informative because the defender moves).
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
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_validation_core as VC
from uav_defend.policies.sanitize import build_policy_info

CURRICULUM = "journal_v2_a1_curriculum"
STAGE = "stage3"
ACTION_VERSION, OBSERVATION_VERSION, REWARD_VERSION = S.ACTION_VERSION, S.OBSERVATION_VERSION, S.REWARD_VERSION
TRAINING_PROTOCOL = "a1_cur3_training_v1"
SCENARIO_PROTOCOL = "a1_cur3_moving_greedy_nominal_path_v1"
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE
STAGE2V2_RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / "stage2_v2"
STAGE2V2_DOCS = "docs/journal_v2_a1_curriculum/stage2_v2"
STAGE2V2_COMMIT = "1214c38"
STAGE2V2_PREFIX = {201: "d70b85a6", 202: "97a934bc", 203: "06993dca"}
PARENT_STATUS = "provisional_curriculum_transfer"

# ----------------------------------------------------------------------------- pre-registered constants (frozen before any training)
SCENARIO_STREAM_TAG = 0xC3A1
VALIDATION_A = (148_000, 148_099)
VALIDATION_B = (148_100, 148_199)
STAGE1_ANCHOR = (151_000, 151_099)        # Stage-1 no-defender pursuit anchor (evaluation only)
STATIC_ANCHOR = (151_100, 151_199)        # frozen Stage-2 static-defender anchor (evaluation only)
DIAGNOSTIC = (152_000, 152_099)           # generalization diagnostic vs Kalman-Greedy / PN / Lead (evaluation only)
SCENARIO_VALIDATION = (153_000, 153_999)  # 1,000 pre-training difficulty-validation scenarios
A0_INTERCEPTION_RANGE = (0.60, 1.00)
N_ENVS, MAX_ROLLOUTS, PPO_SEEDS = S.N_ENVS, 16, S.PPO_SEEDS
STAGE1_ANCHOR_MIN, STATIC_ANCHOR_MIN, ANCHOR_FAILURE_STREAK = 0.98, 0.80, 2
DIAGNOSTIC_CONTROLLERS = ("kalman_greedy", "pn", "lead")
HEADING_BINS = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90.0001))
INTERCEPT_OUTCOMES = S.INTERCEPT_OUTCOMES

RESERVED_RANGES = {**S.RESERVED_RANGES, "stage2_validation": (147_000, 147_199), "stage2_placement_validation": (149_000, 149_999)}
# The Stage-2 anchor (148000-148099, evaluation only) shares NUMERIC seeds with Stage-3 validation set A as pre-registered; it was never used for training, and the scenarios are different tasks
# (static vs moving defender), so only the initial attacker geometry coincides.
KNOWN_NUMERIC_OVERLAP = {"stage2_anchor_eval_only": (148_000, 148_099)}

GATE = {
    "capture_min": 0.80, "improvement_over_a0_min_pp": 20.0, "interception_max": 0.20, "timeout_max": 0.02, "exhaustion_max": 0.02, "median_path_excess_max": 1.80, "gt3_max": 0.05, "gt5_max": 0.0,
    "orbit_episodes_max": 0, "stage1_anchor_capture_min": STAGE1_ANCHOR_MIN, "static_anchor_capture_min": STATIC_ANCHOR_MIN, "nonfinite_max": 0,
}


def seeds_of(rng_tuple) -> range:
    return range(rng_tuple[0], rng_tuple[1] + 1)


def validation_seeds(name: str) -> range:
    return seeds_of({"A": VALIDATION_A, "B": VALIDATION_B}[name])


def final_evaluation_seeds() -> list[int]:
    return list(validation_seeds("A")) + list(validation_seeds("B"))


def namespace_overlaps() -> dict:
    own = {"validation_A": VALIDATION_A, "validation_B": VALIDATION_B, "stage1_anchor": STAGE1_ANCHOR, "static_anchor": STATIC_ANCHOR, "diagnostic": DIAGNOSTIC, "scenario_validation": SCENARIO_VALIDATION}
    out = {}
    for name, (lo, hi) in own.items():
        out[name] = [k for k, (a, b) in RESERVED_RANGES.items() if not (hi < a or lo > b)]
    names = list(own)
    out["mutual"] = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if not (own[a][1] < own[b][0] or own[a][0] > own[b][1])]
    out["training_stream"] = [C.TRAIN_SEED_LOW, C.TRAIN_SEED_HIGH]
    out["documented_numeric_overlap"] = {k: list(v) for k, v in KNOWN_NUMERIC_OVERLAP.items()}
    return out


# ----------------------------------------------------------------------------- environment
def stage3_config(controller: str = "greedy"):
    """Canonical C0 V2 config of the controller's estimator track with the asset frozen and standby synchronisation disabled (see module docstring)."""
    cfg, _estimator = VC.build_config("C0", "V2", controller)
    return dataclasses.replace(cfg, v_s=0.0, defender_standby_until_detection=False)


def estimator_of(controller: str) -> str:
    return "kalman" if controller == "kalman_greedy" else "measurement"


def config_differences_from_canonical() -> dict:
    canonical = A.build_episode_config("C0", "greedy")
    stage = stage3_config("greedy")
    return {f.name: {"canonical": getattr(canonical, f.name), "stage3": getattr(stage, f.name)} for f in dataclasses.fields(canonical) if getattr(canonical, f.name) != getattr(stage, f.name)}


def config_sha256(controller: str = "greedy") -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(stage3_config(controller)), sort_keys=True, default=str).encode()).hexdigest()


def scenario_stream(ppo_seed: int, worker_index: int, rollout_index: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SCENARIO_STREAM_TAG, int(ppo_seed), int(worker_index), int(rollout_index)]))


class Stage3Env(S.Stage2Env):
    """Stage-2 scenario geometry with the stationary defender replaced by a scripted controller that drives the canonical constrained defender dynamics."""

    def __init__(self, *args, controller: str = "greedy", **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = controller
        self._config = stage3_config(controller)
        self._policy = None
        self._def_pos: list = []
        self._def_vel: list = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)         # Stage-1 geometry + Stage-2 placement; defender at rest at the placed position
        self._policy = VC.make_policy(self.controller, self._config)
        self._policy.reset()
        env = self._env
        self._def_pos = [np.asarray(env._defender_pos, dtype=np.float64).copy()]
        self._def_vel = [np.asarray(env._defender_vel, dtype=np.float64).copy()]
        return obs, info

    def _defender_action(self) -> np.ndarray:
        env = self._env
        return np.asarray(self._policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator_of(self.controller))), dtype=np.float32).reshape(3)

    def step(self, action):
        env = self._env
        original = env.step

        def with_defender(_zero):                                      # Stage1Env.step feeds the (ignored) parked-defender zero action; the scripted controller acts instead
            out = original(self._defender_action())
            self._def_pos.append(np.asarray(env._defender_pos, dtype=np.float64).copy())
            self._def_vel.append(np.asarray(env._defender_vel, dtype=np.float64).copy())
            return out

        env.step = with_defender
        try:
            return super(S.Stage2Env, self).step(action)               # Stage1Env.step (not Stage2's static variant)
        finally:
            del env.step

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        out = C.Stage1Env._episode_summary(self, outcome, path_now, range_next)
        att = np.vstack([self._start_pos[None, :], np.array(self._att)])
        dpos = np.array(self._def_pos[: len(att)])
        dvel = np.array(self._def_vel[: len(att)])
        dist = np.linalg.norm(att - dpos, axis=1)
        visible = dist <= self._detection_range
        out.update(
            defender_position=[float(x) for x in dpos[0]], defender_final_position=[float(x) for x in dpos[-1]], placement_lambda=self.placement["lambda"], placement_offset_m=self.placement["offset_m"],
            placement_chord_fraction=self.placement["chord_fraction"], nominal_path_length_m=self.placement["nominal_path_length_m"], placement_attempts=self.placement["attempts"],
            defender_initial_distance_m=float(dist[0]), defender_visible_steps=int(visible[1:].sum()), defender_visible_initially=bool(visible[0]),
            defender_first_visible_step=(int(np.argmax(visible)) if visible.any() else None), defender_visible_s=float(visible[1:].sum() * self._dt), intercepted=outcome in INTERCEPT_OUTCOMES,
            defender_max_speed_mps=float(np.linalg.norm(dvel, axis=1).max()), defender_controller=self.controller,
            defender_speed_at_first_visibility_mps=(float(np.linalg.norm(dvel[int(np.argmax(visible))])) if visible.any() else None),
        )
        return out


def make_env(ppo_seed: int = 0, worker_index: int = 0, rollout_index: int = 0, **kwargs) -> Stage3Env:
    return Stage3Env(ppo_seed, worker_index, rollout_index, **kwargs)


# ----------------------------------------------------------------------------- evaluation / metrics / gate / decision
def evaluate(act, seeds, *, scripted_a0: bool = False, record_trajectory: bool = False, controller: str = "greedy") -> list[dict]:
    env = Stage3Env(0, 0, 0, scripted_a0=scripted_a0, record_trajectory=record_trajectory, controller=controller)
    return [C.run_episode(env, s, None if scripted_a0 else act) for s in seeds]


def metrics(episodes: list[dict]) -> dict:
    return S.metrics(episodes)


def gate_status(m: dict, *, a0_capture: float, stage1_anchor: float, static_anchor: float) -> dict:
    improvement_pp = round(100.0 * (m["capture"] - a0_capture), 9)
    checks = {
        "capture": m["capture"] >= GATE["capture_min"], "improvement_over_a0": improvement_pp >= GATE["improvement_over_a0_min_pp"], "interception": m["interception"] <= GATE["interception_max"],
        "timeout": m["timeout"] <= GATE["timeout_max"], "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"], "median_path_excess": m["median_path_excess"] <= GATE["median_path_excess_max"],
        "gt3": m["gt3"] <= GATE["gt3_max"], "gt5": m["gt5"] <= GATE["gt5_max"], "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"],
        "stage1_anchor_capture": stage1_anchor >= GATE["stage1_anchor_capture_min"], "static_anchor_capture": static_anchor >= GATE["static_anchor_capture_min"],
        "numerical_failures": m["nonfinite"] <= GATE["nonfinite_max"],
    }
    return {"checks": checks, "improvement_over_a0_pp": improvement_pp, "passed": bool(all(checks.values()))}


def streak_failure_rollout(values: list[float], threshold: float, streak: int = ANCHOR_FAILURE_STREAK) -> int | None:
    """1-based rollout at which `values` has been below `threshold` for `streak` consecutive rollouts, else None."""
    run = 0
    for k, v in enumerate(values, start=1):
        run = run + 1 if v < threshold else 0
        if run >= streak:
            return k
    return None


# ----------------------------------------------------------------------------- scenario difficulty validation (canonical A0 vs the moving Greedy defender; no training involved)
def summarize_scenarios(episodes: list[dict], dt: float) -> dict:
    n = len(episodes)

    def dist(values):
        v = np.asarray(values, dtype=np.float64)
        return {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    outcome = np.array([e["outcome"] for e in episodes])
    icpt = np.isin(outcome, INTERCEPT_OUTCOMES)
    off = np.array([e["initial_heading_offset_deg"] for e in episodes])
    bins = {f"{lo}-{min(hi, 90)}": {"n": int(((off >= lo) & (off < hi)).sum()), "a0_interception": float(icpt[(off >= lo) & (off < hi)].mean()) if ((off >= lo) & (off < hi)).any() else None,
                                      "a0_capture": float((outcome[(off >= lo) & (off < hi)] == "soldier_caught").mean()) if ((off >= lo) & (off < hi)).any() else None} for lo, hi in HEADING_BINS}
    steps = np.array([e["steps"] for e in episodes])
    vis_before = [e["defender_visible_s"] for e in episodes if e["intercepted"]]
    first_vis = [e["defender_first_visible_step"] for e in episodes if e["intercepted"] and e["defender_first_visible_step"] is not None]
    return {
        "scenarios": n, "a0_interception_fraction": float(icpt.mean()), "a0_capture_fraction": float((outcome == "soldier_caught").mean()),
        "a0_other_fraction": float(1.0 - icpt.mean() - (outcome == "soldier_caught").mean()), "a0_outcomes": {k: int((outcome == k).sum()) for k in sorted(set(outcome.tolist()))},
        "a0_episode_duration_s": dist(steps * dt), "a0_defender_visible_s_before_interception": dist(vis_before) if vis_before else None,
        "a0_first_visible_step_of_intercepted": dist(first_vis) if first_vis else None, "defender_visible_at_start_fraction": float(np.mean([e["defender_visible_initially"] for e in episodes])),
        "a0_min_defender_separation_m": dist([e["min_defender_separation_m"] for e in episodes]), "heading_bin_breakdown": bins,
        "attacker_asset_distance_m": dist([e["initial_range_m"] for e in episodes]), "defender_along_track_fraction": dist([e["placement_lambda"] for e in episodes]),
        "defender_perpendicular_offset_m": dist([e["placement_offset_m"] for e in episodes]), "defender_initial_distance_to_attacker_m": dist([e["defender_initial_distance_m"] for e in episodes]),
        "defender_max_speed_mps": dist([e["defender_max_speed_mps"] for e in episodes]), "defender_speed_at_first_visibility_mps": dist([e["defender_speed_at_first_visibility_mps"] for e in episodes
                                                                                                                                      if e["defender_speed_at_first_visibility_mps"] is not None]),
        "placement_attempts_max": int(max(e["placement_attempts"] for e in episodes)), "nonfinite_episodes": int(sum(not e["finite"] for e in episodes)),
        "immediate_contact_episodes_steps_le_1": int((steps <= 1).sum()), "min_steps": int(steps.min()),
    }


def scenario_validation(seeds=None) -> dict:
    seeds = list(seeds if seeds is not None else seeds_of(SCENARIO_VALIDATION))
    return summarize_scenarios(evaluate(None, seeds, scripted_a0=True), float(stage3_config().dt))


def protocol_specification(parents: dict | None = None) -> dict:
    return {
        "hypothesis": "A policy that has learned static-threat evasion can transfer that behavior to a moving scripted interceptor while preserving its expert-initialized pursuit capability.",
        "tests": "moving-adversary complexity only (asset stationary, C0, no obstacles, no defender-policy diversity)",
        "identifiers": {"training_protocol": TRAINING_PROTOCOL, "scenario_protocol": SCENARIO_PROTOCOL, "action": ACTION_VERSION, "observation": OBSERVATION_VERSION, "reward": REWARD_VERSION,
                        "curriculum": CURRICULUM, "stage": STAGE},
        "parents": {"stage2_v2_commit": STAGE2V2_COMMIT, "status": PARENT_STATUS,
                    "note": "Stage2-v2 did NOT satisfy its frozen mastery rule (0/3); the rollout-16 checkpoints are PROVISIONAL CURRICULUM INITIALIZATION CHECKPOINTS after manual inspection, not accepted/final A1 models",
                    "checkpoints": parents or {}},
        "environment": {"clutter": "C0 (no static or dynamic obstacles)", "asset": "stationary (v_s = 0)", "config": "canonical C0 V2 measurement-track config with v_s=0 and defender_standby_until_detection=False",
                        "env_config_sha256": config_sha256(), "differences_from_canonical": config_differences_from_canonical(),
                        "defender": "canonical scripted `greedy` controller (unmodified): escorts the asset until it detects the attacker within detection_radius, then pure-pursues the measured attacker position; "
                                    "canonical constrained defender dynamics (18 m/s, 8 m/s^2, 90 deg/s); canonical swept interception 2.5 m; starts AT REST at the placed position",
                        "placement": f"Stage2-v2 nominal-A0-path placement ({S.PLACEMENT_PROTOCOL}): lam ~ U(0.35,0.65), r ~ U(0,1.5 m), never resampled to force an outcome",
                        "defender_observation": "canonical 25 m true-range gate (flag, position, velocity features)"},
        "initialization": {"actor_critic_log_std": "policy state_dict of the matching Stage2-v2 rollout-16 model (actor, CRITIC and log_std preserved: observation, reward and value semantics are unchanged)",
                           "optimizer": "RESET (fresh Adam from PPO construction) at the curriculum boundary", "timestep_counter": "reset to 0 (Stage-3 rollouts are numbered 1..16)",
                           "not_loaded": ["BC checkpoints", "failed PPO models", "Stage-2 optimizer state"]},
        "reward": REWARD_VERSION + " unchanged (no evasion/distance/direction/action terms)",
        "ppo": {"policy": A.A1_V2_POLICY, "hyperparameters": A.A1_V2_PPO_HYPERPARAMS, "n_envs": N_ENVS, "rollout_steps": N_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"], "max_rollouts": MAX_ROLLOUTS,
                "max_timesteps": MAX_ROLLOUTS * N_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"], "seeds": list(PPO_SEEDS)},
        "validation": {"A": list(VALIDATION_A), "B": list(VALIDATION_B), "alternation": "rollout k: odd -> A, even -> B", "stage1_anchor": list(STAGE1_ANCHOR), "static_anchor": list(STATIC_ANCHOR),
                       "diagnostic": list(DIAGNOSTIC), "scenario_validation": list(SCENARIO_VALIDATION), "final": "148000-148199 (200 episodes) + matched canonical A0", "namespace_overlaps": namespace_overlaps()},
        "scenario_validation_rule": f"A0 interception in {A0_INTERCEPTION_RANGE} on {SCENARIO_VALIDATION}; < 0.60 => too easy => STOP; invalid states / degenerate immediate contact => STOP; no placement tuning after PPO results",
        "gate": GATE, "mastery": "two consecutive rollout validations pass ALL gate checks",
        "retention": {"stage1_anchor": f"capture < {STAGE1_ANCHOR_MIN} on {ANCHOR_FAILURE_STREAK} consecutive rollouts => PURSUIT_RETENTION_FAILURE",
                      "static_anchor": f"capture < {STATIC_ANCHOR_MIN} on {ANCHOR_FAILURE_STREAK} consecutive rollouts => STATIC_EVASION_RETENTION_FAILURE"},
        "generalization_diagnostic": {"controllers": list(DIAGNOSTIC_CONTROLLERS), "seeds": list(DIAGNOSTIC), "note": "diagnostic only, never used for mastery or training"},
        "success_rule": ">= 2/3 seeds master, mastered seeds retain Stage-1 pursuit and static-defender evasion, no shared structural pathology",
        "paired_statistics": {"bootstrap": "paired percentile bootstrap over scenarios", "resamples": S.BOOTSTRAP_RESAMPLES, "seed": S.BOOTSTRAP_SEED},
    }


def protocol_sha256(parents: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(protocol_specification(parents), sort_keys=True, default=str).encode()).hexdigest()
