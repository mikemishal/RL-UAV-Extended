"""Curriculum Stage3-v2 (`a1_curriculum_stage3_v2`) for the V2 learned attacker A1: evade a MOVING scripted Greedy defender that is HELD STATIONARY until canonical detection, then released with the unchanged
Greedy controller; stationary asset, C0, no obstacles.

Stage3-v1 (`journal_v2_a1_cur3.py`, kept verbatim as design-failure evidence) let the canonical Greedy escort the asset from step 0. Placed on A0's nominal path, the defender therefore accelerated AWAY from
the approaching attacker (towards the asset) and held 8-14 m/s of wrong-direction velocity at detection (15 m) -> canonical A0 was intercepted in 0/1000 scenarios and the task could not teach evasion.

The ONE conceptual change of Stage3-v2: defender motion is suppressed before canonical detection; after detection the existing Greedy controller is released unchanged. Curriculum scaffold only, NOT the final
defender benchmark; later stages must remove it.

Hold / release semantics (curriculum wrapper; `uav_defend`, Greedy and A0 are untouched):
  * before detection the defender position is exactly fixed, its velocity and acceleration are zero and the Greedy action is NOT applied (`_move_defender` is gated);
  * detection is the canonical environment criterion `env._enemy_detected` (true 3-D attacker-defender distance <= `config.detection_radius` = 15 m, evaluated in `SoldierEnv._update_detection`, sticky);
  * the step in which detection becomes true still has the defender held (same ordering as the canonical standby contract: the first executed controller action is the NEXT step, computed from the first observation
    with `enemy_detected = True`); from then on the unmodified Greedy action drives the unmodified constrained defender dynamics (18 m/s, 8 m/s^2, 90 deg/s) and the defender never returns to hold;
  * release is environment-side curriculum state only: it is not part of the 37-D observation (see `docs/.../stage3_v2/README.md` for the hidden-state audit).
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
from experiments import journal_v2_a1_cur3 as V1
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_validation_core as VC
from uav_defend.policies.sanitize import build_policy_info

VERSION = "a1_curriculum_stage3_v2"
CURRICULUM = "journal_v2_a1_curriculum"
STAGE = "stage3_v2"
ACTION_VERSION, OBSERVATION_VERSION, REWARD_VERSION = S.ACTION_VERSION, S.OBSERVATION_VERSION, S.REWARD_VERSION
TRAINING_PROTOCOL = "a1_cur3v2_training_v1"
SCENARIO_PROTOCOL = "a1_cur3v2_hold_until_detection_greedy_v1"
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE
DOCS = "docs/journal_v2_a1_curriculum/stage3_v2/"
V1_DOCS = "docs/journal_v2_a1_curriculum/stage3/"
STAGE2V2_RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / "stage2_v2"
STAGE2V2_COMMIT = "1214c38"
STAGE2V2_PREFIX = V1.STAGE2V2_PREFIX
PARENT_STATUS = "provisional_curriculum_transfer"
PARENT_ROLLOUT = 16

# ----------------------------------------------------------------------------- pre-registered constants (frozen before any Stage3-v2 training)
SCENARIO_STREAM_TAG = 0xC3B1
# The suggested 149000-149199 conflicts with Stage 2's own placement-validation namespace (149000-149999, evaluation only), so - as the decision allows - a FRESH block of unreserved seeds is used.
VALIDATION_A = (154_000, 154_099)
VALIDATION_B = (154_100, 154_199)
STAGE1_ANCHOR = (155_000, 155_099)
STATIC_ANCHOR = (155_100, 155_199)
DIAGNOSTIC = (156_000, 156_099)
SCENARIO_VALIDATION = (157_000, 157_999)
A0_INTERCEPTION_RANGE = (0.70, 1.00)
MIN_VISIBLE_DECISIONS = 2     # an intercepted episode must have given the policy >= 2 decisions with the defender inside the 25 m observation gate, else contact is unavoidable-by-design
N_ENVS, MAX_ROLLOUTS, PPO_SEEDS = S.N_ENVS, 16, S.PPO_SEEDS
BC_SEEDS = S.BC_SEEDS
DIAGNOSTIC_CONTROLLERS = V1.DIAGNOSTIC_CONTROLLERS
INTERCEPT_OUTCOMES = S.INTERCEPT_OUTCOMES
HEADING_BINS = V1.HEADING_BINS
GATE = dict(V1.GATE)                                     # identical to the Stage3-v1 gate
STAGE1_ANCHOR_MIN, STATIC_ANCHOR_MIN, ANCHOR_FAILURE_STREAK = V1.STAGE1_ANCHOR_MIN, V1.STATIC_ANCHOR_MIN, V1.ANCHOR_FAILURE_STREAK
gate_status, streak_failure_rollout = V1.gate_status, V1.streak_failure_rollout
RESERVED_RANGES = {**V1.RESERVED_RANGES, "stage3_v1_validation_A": V1.VALIDATION_A, "stage3_v1_validation_B": V1.VALIDATION_B, "stage3_v1_stage1_anchor": V1.STAGE1_ANCHOR,
                   "stage3_v1_static_anchor": V1.STATIC_ANCHOR, "stage3_v1_diagnostic": V1.DIAGNOSTIC, "stage3_v1_scenario_validation": V1.SCENARIO_VALIDATION}
HOLD_DYNAMICS_INFO = {"accel_used": 0.0, "turn_rate_used_deg": 0.0, "accel_saturated": False, "turn_saturated": False, "climb_saturated": False}


def seeds_of(rng_tuple) -> range:
    return range(rng_tuple[0], rng_tuple[1] + 1)


def validation_seeds(name: str) -> range:
    return seeds_of({"A": VALIDATION_A, "B": VALIDATION_B}[name])


def final_evaluation_seeds() -> list[int]:
    return list(validation_seeds("A")) + list(validation_seeds("B"))


def validation_set_for(k: int) -> str:
    return "A" if k % 2 == 1 else "B"


def namespace_overlaps() -> dict:
    own = {"validation_A": VALIDATION_A, "validation_B": VALIDATION_B, "stage1_anchor": STAGE1_ANCHOR, "static_anchor": STATIC_ANCHOR, "diagnostic": DIAGNOSTIC, "scenario_validation": SCENARIO_VALIDATION}
    out = {name: [k for k, (a, b) in RESERVED_RANGES.items() if not (hi < a or lo > b)] for name, (lo, hi) in own.items()}
    names = list(own)
    out["mutual"] = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if not (own[a][1] < own[b][0] or own[a][0] > own[b][1])]
    out["training_stream"] = [C.TRAIN_SEED_LOW, C.TRAIN_SEED_HIGH]
    out["training_overlap"] = [n for n, (lo, hi) in own.items() if not (hi < C.TRAIN_SEED_LOW or lo >= C.TRAIN_SEED_HIGH)]
    out["rejected_suggestion_149000_149199"] = "overlaps Stage-2 placement-validation 149000-149999 (evaluation only); fresh block used"
    return out


# ----------------------------------------------------------------------------- configuration
def stage3v2_config(controller: str = "greedy"):
    """Canonical C0 V2 config of the controller's estimator track; asset frozen (v_s = 0) and standby synchronisation disabled (the placed defender must not be re-synchronised onto the asset). Identical to
    Stage3-v1; the hold is applied by the wrapper, not by a config field."""
    return V1.stage3_config(controller)


def config_sha256(controller: str = "greedy") -> str:
    return V1.config_sha256(controller)


def detection_radius(controller: str = "greedy") -> float:
    return float(stage3v2_config(controller).detection_radius)


def scenario_stream(ppo_seed: int, worker_index: int, rollout_index: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SCENARIO_STREAM_TAG, int(ppo_seed), int(worker_index), int(rollout_index)]))


# ----------------------------------------------------------------------------- environment
class Stage3V2Env(S.Stage2Env):
    """Stage-2 scenario geometry (Stage-1 attacker/asset + nominal-A0-path defender placement) with a scripted defender that is held until canonical detection and then driven by the unchanged controller.
    `hold_until_detection=False` reproduces the Stage3-v1 escort behaviour for the v1-vs-v2 comparison only."""

    def __init__(self, ppo_seed: int, worker_index: int = 0, rollout_index: int = 0, scripted_a0: bool = False, record_trajectory: bool = False, gamma: float = R.GAMMA, *,
                 controller: str = "greedy", hold_until_detection: bool = True):
        super().__init__(ppo_seed, worker_index, rollout_index, scripted_a0, record_trajectory, gamma)
        self.controller, self.hold_until_detection = controller, bool(hold_until_detection)
        self._config = stage3v2_config(controller)
        self._stream = scenario_stream(ppo_seed, worker_index, rollout_index)
        self._policy = None
        self._released, self._release_step, self._detect_step = False, None, None
        self._def_action = np.zeros(3, dtype=np.float32)
        self._def_pos: list = []
        self._def_vel: list = []

    def reseed(self, rollout_index: int) -> int:
        self._stream = scenario_stream(self.ppo_seed, self.worker_index, rollout_index)
        return int(rollout_index)

    @property
    def released(self) -> bool:
        return self._released

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        out = super().reset(seed=seed, options=options)               # Stage-1 geometry + nominal-path placement; the defender is at rest at the placed position
        env = self._env
        self._policy = VC.make_policy(self.controller, self._config)
        self._policy.reset()
        self._released = not self.hold_until_detection
        self._release_step = 0 if self._released else None
        self._detect_step = None
        self._def_action = np.zeros(3, dtype=np.float32)
        self._def_pos = [np.asarray(env._defender_pos, dtype=np.float64).copy()]
        self._def_vel = [np.asarray(env._defender_vel, dtype=np.float64).copy()]
        real_move = type(env)._move_defender.__get__(env)

        def gated_move(_ignored_zero_action):
            if env._enemy_detected and self._detect_step is None:
                self._detect_step = self._steps + 1                    # the env step in progress (Stage1Env increments after env.step returns)
            if self._released:
                real_move(self._def_action)
            else:                                                       # held: position untouched, velocity / acceleration / command exactly zero
                env._defender_vel = np.zeros(3, dtype=np.float32)
                env._defender_desired_command = np.zeros(3, dtype=np.float32)
                env._defender_dynamics_info = dict(HOLD_DYNAMICS_INFO)

        env._move_defender = gated_move
        return out

    def _greedy_action(self) -> np.ndarray:
        env = self._env
        return np.asarray(self._policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator_of(self.controller))), dtype=np.float32).reshape(3)

    def step(self, action):
        env = self._env
        if not self._released and env._enemy_detected:                  # detection was observed at the end of the previous step: release now (never reverts)
            self._released, self._release_step = True, self._steps
        self._def_action = self._greedy_action() if self._released else np.zeros(3, dtype=np.float32)
        result = C.Stage1Env.step(self, action)
        self._def_pos.append(np.asarray(env._defender_pos, dtype=np.float64).copy())
        self._def_vel.append(np.asarray(env._defender_vel, dtype=np.float64).copy())
        return result

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        out = C.Stage1Env._episode_summary(self, outcome, path_now, range_next)
        env = self._env
        att = np.vstack([self._start_pos[None, :], np.array(self._att)])
        dpos = np.vstack([np.array(self._def_pos), np.asarray(env._defender_pos, dtype=np.float64)[None, :]])
        dvel = np.vstack([np.array(self._def_vel), np.asarray(env._defender_vel, dtype=np.float64)[None, :]])
        steps = self._steps
        dist = np.linalg.norm(att - dpos, axis=1)
        visible = dist <= self._detection_range
        speed = np.linalg.norm(dvel, axis=1)
        detect = self._detect_step if self._detect_step is not None else (steps if env._enemy_detected else None)
        release = self._release_step
        closest = int(np.argmin(dist))
        moved_before = float(np.linalg.norm(dpos[: (release if release is not None else steps) + 1] - dpos[0], axis=1).max()) if self.hold_until_detection else float("nan")
        speed_at_detect = float(speed[detect]) if detect is not None else None
        toward = None
        if detect is not None:
            los = att[detect] - dpos[detect]
            toward = float(np.dot(dvel[detect], los / max(np.linalg.norm(los), 1e-9)))
        out.update(
            defender_position=[float(x) for x in dpos[0]], defender_final_position=[float(x) for x in dpos[-1]], placement_lambda=self.placement["lambda"], placement_offset_m=self.placement["offset_m"],
            placement_chord_fraction=self.placement["chord_fraction"], nominal_path_length_m=self.placement["nominal_path_length_m"], placement_attempts=self.placement["attempts"],
            defender_initial_distance_m=float(dist[0]), defender_visible_steps=int(visible[1:].sum()), defender_visible_initially=bool(visible[0]),
            defender_first_visible_step=(int(np.argmax(visible)) if visible.any() else None), defender_visible_s=float(visible[1:].sum() * self._dt), intercepted=outcome in INTERCEPT_OUTCOMES,
            defender_max_speed_mps=float(speed.max()), defender_controller=self.controller, hold_until_detection=self.hold_until_detection,
            detected=detect is not None, detection_step=detect, detection_time_s=(detect * self._dt if detect is not None else None),
            released=bool(self._released), release_step=release, release_time_s=(release * self._dt if release is not None else None),
            first_controller_step=(release + 1 if release is not None else None),
            defender_speed_at_detection_mps=speed_at_detect, defender_vel_toward_attacker_at_detection_mps=toward,
            defender_speed_at_release_mps=(float(speed[release]) if release is not None and release < len(speed) else None),
            defender_moved_before_release_m=moved_before,
            defender_moved_after_release=bool(release is not None and steps > release and float(np.linalg.norm(dpos[-1] - dpos[release])) > 0.0),
            time_release_to_end_s=((steps - release) * self._dt if release is not None else None), closest_step=closest, closest_time_s=closest * self._dt,
            time_detection_to_closest_s=((closest - detect) * self._dt if detect is not None else None),
            defender_speed_at_closest_mps=float(speed[closest]), interception_step_minus_release=(steps - release if (outcome in INTERCEPT_OUTCOMES and release is not None) else None),
        )
        return out


def estimator_of(controller: str) -> str:
    return V1.estimator_of(controller)


def make_env(ppo_seed: int = 0, worker_index: int = 0, rollout_index: int = 0, **kwargs) -> Stage3V2Env:
    return Stage3V2Env(ppo_seed, worker_index, rollout_index, **kwargs)


# ----------------------------------------------------------------------------- evaluation
def evaluate(act, seeds, *, scripted_a0: bool = False, record_trajectory: bool = False, controller: str = "greedy", hold_until_detection: bool = True) -> list[dict]:
    env = Stage3V2Env(0, 0, 0, scripted_a0=scripted_a0, record_trajectory=record_trajectory, controller=controller, hold_until_detection=hold_until_detection)
    return [C.run_episode(env, s, None if scripted_a0 else act) for s in seeds]


def evaluate_stage1_anchor(act, seeds=None) -> list[dict]:
    """Stage-1 no-defender pursuit anchor (deterministic policy mean)."""
    return V1V2.evaluate(act, list(seeds if seeds is not None else seeds_of(STAGE1_ANCHOR)))


def evaluate_static_anchor(act, seeds=None) -> list[dict]:
    """Frozen Stage-2 static-defender anchor: the unchanged Stage-2 environment (stationary active defender on A0's nominal path) on a fresh seed block."""
    return S.evaluate(act, list(seeds if seeds is not None else seeds_of(STATIC_ANCHOR)))


def metrics(episodes: list[dict]) -> dict:
    out = S.metrics(episodes)
    steps_after = [e["time_release_to_end_s"] for e in episodes if e.get("time_release_to_end_s") is not None]
    out["detected_fraction"] = float(np.mean([bool(e.get("detected")) for e in episodes]))
    out["time_release_to_end_s_mean"] = float(np.mean(steps_after)) if steps_after else None
    return out


# ----------------------------------------------------------------------------- scenario validation (A0 vs the defender; no training)
def _dist(values) -> dict | None:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    return {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def summarize_scenarios(episodes: list[dict], dt: float) -> dict:
    n = len(episodes)
    outcome = np.array([e["outcome"] for e in episodes])
    icpt = np.isin(outcome, INTERCEPT_OUTCOMES)
    cap = outcome == "soldier_caught"
    off = np.array([e["initial_heading_offset_deg"] for e in episodes])
    bins = {}
    for lo, hi in HEADING_BINS:
        m = (off >= lo) & (off < hi)
        bins[f"{lo}-{min(hi, 90)}"] = {"n": int(m.sum()), "a0_interception": float(icpt[m].mean()) if m.any() else None, "a0_capture": float(cap[m].mean()) if m.any() else None}
    steps = np.array([e["steps"] for e in episodes])
    detected = [e for e in episodes if e["detected"]]
    captured_eps = [e for e in episodes if e["outcome"] == "soldier_caught"]
    intercepted_eps = [e for e in episodes if e["intercepted"]]
    released_moving = [e for e in episodes if e["released"] and e["time_release_to_end_s"] is not None and e["time_release_to_end_s"] > 0]
    return {
        "scenarios": n, "a0_interception_fraction": float(icpt.mean()), "a0_capture_fraction": float(cap.mean()), "a0_timeout_fraction": float((outcome == "timeout").mean()),
        "a0_other_fraction": float(1.0 - icpt.mean() - cap.mean()), "a0_outcomes": {k: int((outcome == k).sum()) for k in sorted(set(outcome.tolist()))},
        "a0_episode_duration_s": _dist(steps * dt), "detected_fraction": float(len(detected) / n), "detection_time_s": _dist([e["detection_time_s"] for e in episodes]),
        "release_time_s": _dist([e["release_time_s"] for e in episodes]), "first_controller_action_time_s": _dist([e["first_controller_step"] * dt if e["first_controller_step"] is not None else None for e in episodes]),
        "time_release_to_interception_s": _dist([e["time_release_to_end_s"] for e in intercepted_eps]), "time_release_to_capture_s": _dist([e["time_release_to_end_s"] for e in captured_eps]),
        "interception_steps_after_release": _dist([e["interception_step_minus_release"] for e in intercepted_eps]),
        "defender_speed_at_release_mps": _dist([e["defender_speed_at_release_mps"] for e in episodes]),
        "defender_speed_at_detection_mps": _dist([e["defender_speed_at_detection_mps"] for e in episodes]),
        "defender_vel_toward_attacker_at_detection_mps": _dist([e["defender_vel_toward_attacker_at_detection_mps"] for e in episodes]),
        "defender_speed_at_closest_approach_mps": _dist([e["defender_speed_at_closest_mps"] for e in episodes]), "defender_max_speed_mps": _dist([e["defender_max_speed_mps"] for e in episodes]),
        "a0_min_defender_separation_m": _dist([e["min_defender_separation_m"] for e in episodes]), "heading_bin_breakdown": bins,
        "attacker_asset_distance_m": _dist([e["initial_range_m"] for e in episodes]), "defender_along_track_fraction": _dist([e["placement_lambda"] for e in episodes]),
        "defender_perpendicular_offset_m": _dist([e["placement_offset_m"] for e in episodes]), "defender_initial_distance_to_attacker_m": _dist([e["defender_initial_distance_m"] for e in episodes]),
        "defender_visible_at_start_fraction": float(np.mean([e["defender_visible_initially"] for e in episodes])), "placement_attempts_max": int(max(e["placement_attempts"] for e in episodes)),
        "validity": {
            "nonfinite_episodes": int(sum(not e["finite"] for e in episodes)), "episodes_steps_le_1": int((steps <= 1).sum()), "min_steps": int(steps.min()),
            "interceptions_within_1_step_of_release_informational": int(sum(1 for e in intercepted_eps if e["interception_step_minus_release"] is not None and e["interception_step_minus_release"] <= 1)),
            "policy_decisions_with_defender_visible_before_interception": _dist([e["steps"] - e["defender_first_visible_step"] for e in intercepted_eps if e["defender_first_visible_step"] is not None]),
            "intercepted_with_fewer_than_2_visible_decisions": int(sum(1 for e in intercepted_eps if e["defender_first_visible_step"] is None or e["steps"] - e["defender_first_visible_step"] < MIN_VISIBLE_DECISIONS)),
            "defender_initial_distance_min_m": float(min(e["defender_initial_distance_m"] for e in episodes)),
            "released_episodes": int(sum(e["released"] for e in episodes)), "max_defender_speed_at_release_mps": float(max(e["defender_speed_at_release_mps"] or 0.0 for e in episodes)),
            "max_defender_displacement_before_release_m": float(max((e["defender_moved_before_release_m"] for e in episodes if not np.isnan(e["defender_moved_before_release_m"])), default=float("nan"))),
            "released_episodes_with_time_after_release": len(released_moving), "released_episodes_where_greedy_moved_defender": int(sum(e["defender_moved_after_release"] for e in released_moving)),
            "episodes_never_detected": int(sum(not e["detected"] for e in episodes)),
        },
    }


def scenario_validation(seeds=None, *, hold_until_detection: bool = True) -> dict:
    seeds = list(seeds if seeds is not None else seeds_of(SCENARIO_VALIDATION))
    return summarize_scenarios(evaluate(None, seeds, scripted_a0=True, hold_until_detection=hold_until_detection), float(stage3v2_config().dt))


def validation_verdict(summary: dict) -> dict:
    v = summary["validity"]
    icpt = summary["a0_interception_fraction"]
    checks = {
        "a0_interception_in_range": A0_INTERCEPTION_RANGE[0] <= icpt <= A0_INTERCEPTION_RANGE[1], "no_nonfinite_state": v["nonfinite_episodes"] == 0,
        "no_immediate_contact_degeneracy": (v["episodes_steps_le_1"] == 0 and v["intercepted_with_fewer_than_2_visible_decisions"] == 0
                                            and v["defender_initial_distance_min_m"] > float(stage3v2_config().detection_radius)),
        "defender_velocity_at_release_zero": v["max_defender_speed_at_release_mps"] == 0.0, "defender_fixed_before_release": v["max_defender_displacement_before_release_m"] == 0.0,
        "greedy_takes_over_after_release": v["released_episodes_where_greedy_moved_defender"] == v["released_episodes_with_time_after_release"],
    }
    return {"checks": checks, "passed": bool(all(checks.values())), "too_easy": bool(icpt < A0_INTERCEPTION_RANGE[0]), "stop_rule": f"A0 interception in {list(A0_INTERCEPTION_RANGE)}; < {A0_INTERCEPTION_RANGE[0]} => STOP"}


def v1_vs_v2_baseline(seeds=None) -> dict:
    """Same scenarios, same code path: escort (Stage3-v1, `hold_until_detection=False`) vs hold-until-detection (Stage3-v2)."""
    seeds = list(seeds if seeds is not None else seeds_of(SCENARIO_VALIDATION))
    out = {}
    for label, hold in (("stage3_v1_escort", False), ("stage3_v2_hold", True)):
        s = summarize_scenarios(evaluate(None, seeds, scripted_a0=True, hold_until_detection=hold), float(stage3v2_config().dt))
        out[label] = {"a0_interception": s["a0_interception_fraction"], "a0_capture": s["a0_capture_fraction"], "a0_timeout": s["a0_timeout_fraction"],
                      "defender_speed_at_detection_mps_median": s["defender_speed_at_detection_mps"]["median"] if s["defender_speed_at_detection_mps"] else None,
                      "defender_speed_at_detection_mps_p95": s["defender_speed_at_detection_mps"]["p95"] if s["defender_speed_at_detection_mps"] else None,
                      "defender_vel_toward_attacker_at_detection_mps_median": s["defender_vel_toward_attacker_at_detection_mps"]["median"] if s["defender_vel_toward_attacker_at_detection_mps"] else None,
                      "median_min_separation_m": s["a0_min_defender_separation_m"]["median"], "detection_time_s_median": s["detection_time_s"]["median"] if s["detection_time_s"] else None,
                      "episode_duration_s_median": s["a0_episode_duration_s"]["median"]}
    return {"seeds": [seeds[0], seeds[-1]], "scenarios": len(seeds), **out}


# ----------------------------------------------------------------------------- parents (provisional curriculum-transfer checkpoints)
def parent_checkpoint(seed: int, project_root: Path | None = None) -> dict:
    """The Stage2-v2 rollout-16 model of `seed`: path + SHA-256 verified against the Stage2-v2 seed manifest, its rollout-16 entry, the committed docs manifest and the pre-registered hash prefix."""
    from experiments import journal_v2_a1_cur1_training as T1

    root = Path(project_root) if project_root is not None else T1.PROJECT_ROOT
    results = (root / "results" / "journal_v2_corrected_semantics" / "a1_curriculum" / "stage2_v2") if project_root is not None else STAGE2V2_RESULTS_ROOT
    manifest = json.loads((results / f"seed_{seed}" / "checkpoints" / "seed_manifest.json").read_text())
    path = results / f"seed_{seed}" / "checkpoints" / "accepted_model.zip"
    sha = T1.T.sha256_file(path)
    entry = next(r for r in manifest["rollouts"] if r["rollout"] == PARENT_ROLLOUT)
    committed = json.loads((root / "docs" / "journal_v2_a1_curriculum" / "stage2_v2" / f"seed_{seed}_manifest.json").read_text())
    ok = (sha == manifest["accepted_model_sha256"] == entry["model_sha256"] == committed["accepted_model_sha256"] and manifest["accepted_rollout"] == PARENT_ROLLOUT
          and sha.startswith(STAGE2V2_PREFIX[int(seed)]))
    if not ok:
        raise RuntimeError(f"Stage2-v2 parent of seed {seed} failed verification (file {sha})")
    return {"seed": int(seed), "attacker_id": f"A1-CUR2V2-{seed}-REVIEW", "path": str(path), "model_sha256": sha, "rollout": PARENT_ROLLOUT, "num_timesteps": entry["num_timesteps"],
            "stage2_v2_training_status": manifest["status"], "parent_status": PARENT_STATUS, "stage2_v2_commit": STAGE2V2_COMMIT}


def parent_hashes() -> dict:
    return {str(s): {"model_sha256": parent_checkpoint(s)["model_sha256"], "rollout": PARENT_ROLLOUT, "status": PARENT_STATUS} for s in PPO_SEEDS}


# ----------------------------------------------------------------------------- protocol
def protocol_specification(parents: dict | None = None) -> dict:
    parents = parents if parents is not None else parent_hashes()
    v1 = V1.protocol_specification()
    return {
        "version": VERSION,
        "hypothesis": "A policy that has learned static-threat evasion can transfer that behavior to a moving scripted interceptor while preserving its expert-initialized pursuit capability.",
        "tests": "moving-adversary complexity only (asset stationary, C0, no obstacles, no defender-policy diversity)",
        "identifiers": {"training_protocol": TRAINING_PROTOCOL, "scenario_protocol": SCENARIO_PROTOCOL, "action": ACTION_VERSION, "observation": OBSERVATION_VERSION, "reward": REWARD_VERSION,
                        "curriculum": CURRICULUM, "stage": STAGE},
        "single_conceptual_change_from_stage3_v1": "Defender motion is suppressed before canonical detection; after detection, the existing Greedy controller is released unchanged.",
        "other_changes_specified_by_the_decision": {"a0_interception_min": [V1.A0_INTERCEPTION_RANGE[0], A0_INTERCEPTION_RANGE[0]], "validation_namespace": "fresh Stage3-v2 held-out seed blocks (below)",
                                                    "scenario_stream_tag": [hex(V1.SCENARIO_STREAM_TAG), hex(SCENARIO_STREAM_TAG)]},
        "stage3_v1": {"verdict": "STOPPED at scenario validation: A0 interception 0/1000 (escort transient)", "evidence": V1_DOCS + "scenario_validation.json", "protocol_sha256": V1.protocol_sha256(),
                      "outputs_never_mixed": "separate results / scratch / docs namespaces; verify_rollout rejects any manifest with a different protocol sha"},
        "parents": {"stage2_v2_commit": STAGE2V2_COMMIT, "status": PARENT_STATUS, "rollout": PARENT_ROLLOUT,
                    "note": "Stage2-v2 did NOT satisfy its frozen mastery rule (0/3); the rollout-16 checkpoints are PROVISIONAL CURRICULUM INITIALIZATION CHECKPOINTS after manual inspection, not accepted/final A1 models",
                    "checkpoints": parents},
        "environment": {
            "clutter": "C0 (no static or dynamic obstacles)", "asset": "stationary (v_s = 0)", "config": "canonical C0 V2 measurement-track config with v_s=0 and defender_standby_until_detection=False",
            "env_config_sha256": config_sha256(), "differences_from_canonical": V1.config_differences_from_canonical(), "detection_radius_m": detection_radius(),
            "defender": "canonical scripted `greedy` controller, unmodified; HELD fixed (zero velocity/acceleration, Greedy action not applied) until canonical detection (`env._enemy_detected`: true 3-D "
                        "attacker-defender distance <= detection_radius, sticky); the step in which detection becomes true is still held (canonical standby ordering); from the next step the unmodified Greedy action "
                        "drives the unmodified constrained dynamics (18 m/s, 8 m/s^2, 90 deg/s) with canonical swept interception 2.5 m; never returns to hold",
            "placement": f"Stage2-v2 nominal-A0-path placement ({S.PLACEMENT_PROTOCOL}): lam ~ U(0.35,0.65), r ~ U(0,1.5 m), never resampled to force an outcome",
            "defender_observation": "canonical 25 m true-range gate (flag, position, velocity features); release flag / release time / lambda / nominal path are NOT observed",
            "curriculum_scaffold": "hold-until-detection is a curriculum simplification (isolates evasion of a moving interceptor from the pre-contact escort transient); NOT the final defender benchmark; later stages must remove it"},
        "observation_audit": "keep 37-D: the defender is invisible (zero features) outside the 25 m gate and stationary until released, the release (15 m detection) happens while it is already inside the gate, and its onset "
                             "is then inferable from the observed defender velocity feature; the only unobservable quantity is the defender's pre-release dwell, which carries no information the attacker could use",
        "initialization": {"actor_critic_log_std": "policy state_dict of the matching Stage2-v2 rollout-16 model (actor, CRITIC and log_std preserved: observation / reward / value semantics unchanged)",
                           "optimizer": "RESET (fresh Adam from PPO construction) at the curriculum boundary", "timestep_counter": "reset to 0 (Stage3-v2 rollouts numbered 1..16)",
                           "not_loaded": ["BC checkpoints", "failed PPO models", "Stage-2 optimizer state"]},
        "reward": REWARD_VERSION + " unchanged (no evasion/distance/direction/action terms)", "ppo": {"policy": A.A1_V2_POLICY, "hyperparameters": A.A1_V2_PPO_HYPERPARAMS, "n_envs": N_ENVS,
                                                                                               "rollout_steps": N_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"],
                                                           "max_rollouts": MAX_ROLLOUTS, "max_timesteps": MAX_ROLLOUTS * N_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"], "seeds": list(PPO_SEEDS)},
        "validation": {"A": list(VALIDATION_A), "B": list(VALIDATION_B), "alternation": "rollout k: odd -> A, even -> B", "stage1_anchor": list(STAGE1_ANCHOR), "static_anchor": list(STATIC_ANCHOR),
                       "diagnostic": list(DIAGNOSTIC), "scenario_validation": list(SCENARIO_VALIDATION), "final": "154000-154199 (200 episodes) + matched canonical A0 + Stage2-v2 parent",
                       "namespace_overlaps": namespace_overlaps()},
        "scenario_validation_rule": {"range": list(A0_INTERCEPTION_RANGE), "interception_below_min": "STOP (too easy)", "validity": "no immediate-contact degeneracy (no episode ending within 1 step of reset, defender initial "
                                     f"distance > detection radius, every intercepted episode gave the policy >= {MIN_VISIBLE_DECISIONS} decisions with the defender inside the 25 m gate), defender velocity at release = 0, "
                                     "defender fixed before release, Greedy takes over after release, no invalid state"},
        "gate": GATE, "gate_identical_to_stage3_v1": GATE == V1.GATE, "mastery": "two consecutive rollout validations pass ALL gate checks",
        "retention": v1["retention"], "generalization_diagnostic": {"controllers": list(DIAGNOSTIC_CONTROLLERS), "seeds": list(DIAGNOSTIC), "semantics": "same hold-until-detection wrapper; diagnostic only"},
        "success_rule": v1["success_rule"], "paired_statistics": v1["paired_statistics"],
    }


def protocol_sha256(parents: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(protocol_specification(parents), sort_keys=True, default=str).encode()).hexdigest()
