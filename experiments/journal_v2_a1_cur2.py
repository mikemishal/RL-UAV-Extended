"""Curriculum Stage 2 for the V2 learned attacker A1: evade a STATIONARY, fully ACTIVE defender while still reaching a stationary asset (C0, no obstacles).

Identifiers: action `a1_cur1_action_v1` (unchanged), observation `a1_v2_obs_v1` (unchanged, canonical defender gate restored), reward `a1_v2_reward_v2` (unchanged),
training protocol `a1_cur2_training_v1`, placement protocol `a1_cur2_static_defender_v1`. See docs/journal_v2_a1_curriculum/stage2/.

Configuration / wrapper only: `uav_defend` is untouched. The environment configuration is EXACTLY the Stage-1-v1 configuration (canonical C0 V2, passive-harness defender without
standby synchronisation, `v_s = 0`); in particular `intercept_radius` (2.5 m) and the swept interception semantics are canonical, so the defender terminates the episode on contact.

  * asset    : stationary at the canonical origin.
  * defender : never acts (zero action every step, zero velocity), placed per scenario relative to the initial attacker->asset segment (below).
  * obs      : `build_a1_v2_observation(env, config.enemy_defender_detection_range = 25 m)`: defender flag/position/velocity are zero unless the TRUE 3-D attacker-defender range <= 25 m.
  * placement: (user-approved amendment after the first chord-based design failed the 70% A0-interception rule: docs/.../stage2/scenario_validation_initial_STOP.json) q = P(lam) + r (cos(phi) e1 + sin(phi) e2),
               P(lam) = the point at fraction lam of the arc length of the NOISE-FREE CANONICAL-A0 NOMINAL PATH from the Stage-2 initial state (simulated without interception), e1/e2 an orthonormal basis of the plane
               perpendicular to the local path tangent, lam ~ U(0.35, 0.65), r ~ U(0, 1.5 m), phi ~ U(0, 2 pi); draws are rejected (re-sampled, never clipped) unless q lies inside the arena |x|,|y| <= L,
               0 <= z <= max_altitude. A pure function of the scenario seed. The interception radius is unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_r2 as R

CURRICULUM = "journal_v2_a1_curriculum"
STAGE = "stage2"
ACTION_VERSION = C.ACTION_VERSION
OBSERVATION_VERSION = C.OBSERVATION_VERSION
REWARD_VERSION = C.REWARD_VERSION
TRAINING_PROTOCOL = "a1_cur2_training_v1"
PLACEMENT_PROTOCOL = "a1_cur2_static_defender_nominal_path_v1"
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE
STAGE1_V2_RESULTS_ROOT = V1V2.RESULTS_ROOT
STAGE1_V2_HANDOFF = "docs/journal_v2_a1_curriculum/stage1_v2/stage2_handoff.json"
STAGE1_V2_TAG = "journal-v2-a1-curriculum-bc-stage1-v1"

# ----------------------------------------------------------------------------- pre-registered constants (frozen before any training)
SCENARIO_STREAM_TAG = 0xC2A1          # training scenario-seed stream
PLACEMENT_TAG = 0xC2A3                # per-scenario defender placement stream
LAMBDA_RANGE = (0.35, 0.65)
OFFSET_MAX_M = 1.5
MAX_PLACEMENT_ATTEMPTS = 50
VALIDATION_A = (147_000, 147_099)
VALIDATION_B = (147_100, 147_199)
ANCHOR = (148_000, 148_099)           # Stage-1 style no-defender pursuit anchor (never used for BC)
PLACEMENT_VALIDATION = (149_000, 149_999)
A0_INTERCEPTION_MIN = 0.70            # scenario-validation stop rule
N_ENVS = C.N_ENVS
MAX_ROLLOUTS = C.MAX_ROLLOUTS
PPO_SEEDS = C.PPO_SEEDS
BC_SEEDS = {201: "A1-BC1-201", 202: "A1-BC1-202", 203: "A1-BC1-203"}
ANCHOR_RETENTION_MIN = 0.98
ANCHOR_FAILURE_STREAK = 2
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_201
INTERCEPT_OUTCOMES = ("intercepted", "unsafe_intercept")

RESERVED_RANGES = {**V1V2.RESERVED_RANGES, "stage1_v2_expert": (160_000, 164_999), "stage1_v2_relabel": (170_000, 170_999)}

GATE = {  # Stage-2 mastery gate (frozen): TWO CONSECUTIVE rollout validations (alternating sets A/B), each satisfying ALL of these
    "capture_min": 0.90, "improvement_over_a0_min_pp": 20.0, "timeout_max": 0.02, "exhaustion_max": 0.02, "median_path_excess_max": 1.60,
    "gt3_max": 0.02, "gt5_max": 0.0, "orbit_episodes_max": 0, "anchor_capture_min": ANCHOR_RETENTION_MIN, "nonfinite_max": 0,
}
VALIDATION_ROLLOUT_SET = C.VALIDATION_ROLLOUT_SET


def validation_seeds(name: str) -> range:
    lo, hi = {"A": VALIDATION_A, "B": VALIDATION_B}[name]
    return range(lo, hi + 1)


def final_evaluation_seeds() -> list[int]:
    return list(validation_seeds("A")) + list(validation_seeds("B"))


def anchor_seeds() -> range:
    return range(ANCHOR[0], ANCHOR[1] + 1)


def placement_validation_seeds() -> range:
    return range(PLACEMENT_VALIDATION[0], PLACEMENT_VALIDATION[1] + 1)


def namespace_overlaps() -> dict:
    out = {}
    for name, (lo, hi) in (("validation_A", VALIDATION_A), ("validation_B", VALIDATION_B), ("anchor", ANCHOR), ("placement_validation", PLACEMENT_VALIDATION)):
        out[name] = [k for k, (a, b) in RESERVED_RANGES.items() if not (hi < a or lo > b)]
    own = {"validation_A": VALIDATION_A, "validation_B": VALIDATION_B, "anchor": ANCHOR, "placement_validation": PLACEMENT_VALIDATION}
    names = list(own)
    out["mutual"] = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if not (own[a][1] < own[b][0] or own[a][0] > own[b][1])]
    out["training_stream"] = [C.TRAIN_SEED_LOW, C.TRAIN_SEED_HIGH]
    return out


def stage2_config():
    """The Stage-1-v1 configuration verbatim (canonical C0 V2, passive-harness defender w/o standby sync, v_s = 0). Interception radius and swept contact are canonical."""
    return C.stage1_config()


def config_differences_from_canonical() -> dict:
    canonical = A.build_episode_config("C0", "greedy")
    stage = stage2_config()
    return {f.name: {"canonical": getattr(canonical, f.name), "stage2": getattr(stage, f.name)} for f in dataclasses.fields(canonical) if getattr(canonical, f.name) != getattr(stage, f.name)}


def config_sha256() -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(stage2_config()), sort_keys=True, default=str).encode()).hexdigest()


def scenario_stream(ppo_seed: int, worker_index: int, rollout_index: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SCENARIO_STREAM_TAG, int(ppo_seed), int(worker_index), int(rollout_index)]))


# ----------------------------------------------------------------------------- defender placement
def _perpendicular_basis(direction) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(direction, dtype=np.float64)
    m = m / np.linalg.norm(m)
    helper = np.array([0.0, 0.0, 1.0]) if abs(m[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(m, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(m, e1)


def _draw_and_validate(scenario_seed: int, anchor_point_fn, *, arena_half: float, max_altitude: float) -> dict:
    """lam ~ U(0.35,0.65), r ~ U(0,1.5 m), phi ~ U(0,2 pi) from the per-scenario stream; draws whose defender position is outside the arena / altitude band are re-sampled, never clipped."""
    rng = np.random.default_rng(np.random.SeedSequence([PLACEMENT_TAG, int(scenario_seed)]))
    for attempt in range(1, MAX_PLACEMENT_ATTEMPTS + 1):
        lam = float(rng.uniform(*LAMBDA_RANGE))
        r = float(rng.uniform(0.0, OFFSET_MAX_M))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        point, tangent = anchor_point_fn(lam)
        e1, e2 = _perpendicular_basis(tangent)
        q = point + r * (math.cos(phi) * e1 + math.sin(phi) * e2)
        if abs(q[0]) <= arena_half and abs(q[1]) <= arena_half and 0.0 <= q[2] <= max_altitude:
            return {"position": q.astype(np.float32), "lambda": lam, "offset_m": r, "phi_rad": phi, "attempts": attempt, "path_point": point, "path_tangent": tangent / np.linalg.norm(tangent)}
    raise RuntimeError(f"no valid defender placement for scenario {scenario_seed} after {MAX_PLACEMENT_ATTEMPTS} draws")


def place_defender_on_chord(scenario_seed: int, attacker_start, asset, *, arena_half: float, max_altitude: float) -> dict:
    """SUPERSEDED first design (initial attacker->asset chord); kept only to reproduce `scenario_validation_initial_STOP.json` (A0 interception 51.5% < 70%)."""
    pa, ps = np.asarray(attacker_start, dtype=np.float64), np.asarray(asset, dtype=np.float64)
    axis = ps - pa
    return _draw_and_validate(scenario_seed, lambda lam: (pa + lam * axis, axis), arena_half=arena_half, max_altitude=max_altitude)


def polyline_point(path: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Point at `fraction` of the arc length of a polyline and the unit tangent of the segment containing it."""
    seg = np.diff(path, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    s = float(fraction) * cum[-1]
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    t = (s - cum[i]) / lengths[i] if lengths[i] > 0 else 0.0
    return path[i] + t * seg[i], seg[i] / lengths[i]


_NOMINAL_ENV = None


def nominal_a0_path(scenario_seed: int) -> np.ndarray:
    """Deterministic noise-free canonical-A0 trajectory (attacker positions at every step boundary, exact float64) from the Stage-2 initial state, simulated WITHOUT any interception
    (`intercept_radius = 1e-6`, defender parked far away). A0's motion is independent of the defender in C0, so this is exactly the path A0 flies until it meets the defender."""
    global _NOMINAL_ENV
    if _NOMINAL_ENV is None:
        _NOMINAL_ENV = V1V2.Stage1V2Env(0, 0, 0, scripted_a0=True)
    env = _NOMINAL_ENV
    env.reset(options={"scenario": int(scenario_seed)})
    done, info = False, {}
    while not done:
        _o, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
        done = te or tr
    if info["outcome"] != "soldier_caught":
        raise RuntimeError(f"nominal A0 path of scenario {scenario_seed} ended in {info['outcome']}")
    return np.vstack([np.asarray(env._start_pos, dtype=np.float64)[None, :]] + [np.asarray(p, dtype=np.float64)[None, :] for p in env._att])


def place_defender(scenario_seed: int, nominal_path: np.ndarray, *, arena_half: float, max_altitude: float) -> dict:
    """Deterministic static-defender placement relative to the nominal A0 path (lam along the path's arc length, perpendicular offset r around the local tangent)."""
    out = _draw_and_validate(scenario_seed, lambda lam: polyline_point(nominal_path, lam), arena_half=arena_half, max_altitude=max_altitude)
    chord = nominal_path[-1] - nominal_path[0]
    out["chord_fraction"] = float(np.dot(out["position"].astype(np.float64) - nominal_path[0], chord) / np.dot(chord, chord))
    out["nominal_path_length_m"] = float(np.linalg.norm(np.diff(nominal_path, axis=0), axis=1).sum())
    return out


class Stage2Env(C.Stage1Env):
    """Stage-1-v1 geometry (initial position + random initial heading, stationary asset) with an active, stationary defender placed on A0's nominal path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = stage2_config()
        self.placement: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        env = self._env
        path = nominal_a0_path(self.scenario_seed)
        if not np.allclose(path[0], np.asarray(env._enemy_pos, dtype=np.float64), atol=1e-4):
            raise RuntimeError("nominal path and Stage-2 episode do not start from the same state")
        placement = place_defender(self.scenario_seed, path, arena_half=float(self._config.L), max_altitude=float(self._config.max_altitude))
        env._defender_pos = placement["position"].copy()
        env._defender_vel = np.zeros(3, dtype=np.float32)
        self.placement = placement
        self._defender_fixed = placement["position"].astype(np.float64)
        return self._observation(), self._info({}, None)

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        out = super()._episode_summary(outcome, path_now, range_next)
        positions = np.vstack([self._start_pos[None, :], np.array(self._att)])
        dist = np.linalg.norm(positions - self._defender_fixed[None, :], axis=1)
        visible = dist <= self._detection_range
        out.update(
            defender_position=[float(x) for x in self._defender_fixed], placement_lambda=self.placement["lambda"], placement_offset_m=self.placement["offset_m"],
            placement_chord_fraction=self.placement["chord_fraction"], nominal_path_length_m=self.placement["nominal_path_length_m"],
            placement_attempts=self.placement["attempts"], defender_initial_distance_m=float(dist[0]), defender_visible_steps=int(visible[1:].sum()),
            defender_visible_initially=bool(visible[0]), defender_first_visible_step=(int(np.argmax(visible)) if visible.any() else None),
            defender_visible_s=float(visible[1:].sum() * self._dt), intercepted=outcome in INTERCEPT_OUTCOMES,
        )
        return out


def make_env(ppo_seed: int = 0, worker_index: int = 0, rollout_index: int = 0, **kwargs) -> Stage2Env:
    return Stage2Env(ppo_seed, worker_index, rollout_index, **kwargs)


# ----------------------------------------------------------------------------- evaluation / metrics / gate
def evaluate(act, seeds, *, scripted_a0: bool = False, record_trajectory: bool = False) -> list[dict]:
    env = Stage2Env(0, 0, 0, scripted_a0=scripted_a0, record_trajectory=record_trajectory)
    return [C.run_episode(env, s, None if scripted_a0 else act) for s in seeds]


def evaluate_anchor(act, seeds=None) -> list[dict]:
    """Stage-1 no-defender pursuit anchor (curriculum_basic_pursuit_no_defender_interaction), deterministic policy mean."""
    return V1V2.evaluate(act, list(seeds if seeds is not None else anchor_seeds()))


def metrics(episodes: list[dict]) -> dict:
    out = C.metrics(episodes)
    pe = np.array([e["path_excess"] for e in episodes], dtype=np.float64)
    sep = np.array([e["min_defender_separation_m"] for e in episodes], dtype=np.float64)
    out["interception"] = float(np.mean([e.get("intercepted", e["outcome"] in INTERCEPT_OUTCOMES) for e in episodes]))
    out["gt3"] = float(np.mean(pe > 3.0))
    out["min_defender_separation_median_m"] = float(np.median(sep))
    out["min_defender_separation_p5_m"] = float(np.percentile(sep, 5))
    out["min_defender_separation_mean_m"] = float(np.mean(sep))
    out["defender_visible_s_mean"] = float(np.mean([e.get("defender_visible_s", 0.0) for e in episodes]))
    out["defender_visible_episode_fraction"] = float(np.mean([e.get("defender_visible_steps", 0) > 0 for e in episodes]))
    return out


def gate_status(m: dict, *, a0_capture: float, anchor_capture: float) -> dict:
    improvement_pp = round(100.0 * (m["capture"] - a0_capture), 9)
    checks = {
        "capture": m["capture"] >= GATE["capture_min"], "improvement_over_a0": improvement_pp >= GATE["improvement_over_a0_min_pp"], "timeout": m["timeout"] <= GATE["timeout_max"],
        "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"], "median_path_excess": m["median_path_excess"] <= GATE["median_path_excess_max"], "gt3": m["gt3"] <= GATE["gt3_max"],
        "gt5": m["gt5"] <= GATE["gt5_max"], "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"], "anchor_capture": anchor_capture >= GATE["anchor_capture_min"],
        "numerical_failures": m["nonfinite"] <= GATE["nonfinite_max"],
    }
    return {"checks": checks, "improvement_over_a0_pp": improvement_pp, "passed": bool(all(checks.values()))}


def retention_failure_rollout(anchor_captures: list[float]) -> int | None:
    """1-based rollout at which anchor capture has been below the retention minimum for ANCHOR_FAILURE_STREAK consecutive rollouts, else None."""
    run = 0
    for k, value in enumerate(anchor_captures, start=1):
        run = run + 1 if value < ANCHOR_RETENTION_MIN else 0
        if run >= ANCHOR_FAILURE_STREAK:
            return k
    return None


def paired_bootstrap(a1_success, a0_success, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Matched-scenario difference (A1 - A0) of a 0/1 outcome with a paired percentile bootstrap 95% CI (resampling scenarios)."""
    a, b = np.asarray(a1_success, dtype=np.float64), np.asarray(a0_success, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired outcomes must be 1-D arrays of equal length")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(len(diff)), "a1_rate": float(a.mean()), "a0_rate": float(b.mean()), "difference_pp": float(100.0 * diff.mean()), "ci95_pp": [float(100.0 * lo), float(100.0 * hi)],
            "a1_only": int(((a == 1) & (b == 0)).sum()), "a0_only": int(((a == 0) & (b == 1)).sum()), "resamples": int(resamples), "bootstrap_seed": int(seed)}


# ----------------------------------------------------------------------------- scenario validation (A0 before any training)
def scenario_validation(seeds=None) -> dict:
    """Canonical A0 on the deterministic Stage-2 scenarios (no training involved): placement distribution, interception/capture fractions, gate-entry lead time."""
    seeds = list(seeds if seeds is not None else placement_validation_seeds())
    episodes = evaluate(None, seeds, scripted_a0=True)
    return summarize_scenarios(episodes, float(stage2_config().dt))


def summarize_scenarios(episodes: list[dict], dt: float) -> dict:
    n = len(episodes)

    def dist(values):
        v = np.asarray(values, dtype=np.float64)
        return {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    outcome = np.array([e["outcome"] for e in episodes])
    intercepted = np.isin(outcome, INTERCEPT_OUTCOMES)
    lead = [(e["steps"] - e["defender_first_visible_step"]) * dt for e in episodes if e["intercepted"] and e["defender_first_visible_step"] is not None]
    return {
        "scenarios": n, "attacker_asset_distance_m": dist([e["initial_range_m"] for e in episodes]), "defender_along_track_fraction_of_nominal_path": dist([e["placement_lambda"] for e in episodes]),
        "defender_projected_chord_fraction": dist([e["placement_chord_fraction"] for e in episodes]), "nominal_a0_path_length_m": dist([e["nominal_path_length_m"] for e in episodes]),
        "defender_perpendicular_offset_m": dist([e["placement_offset_m"] for e in episodes]), "defender_altitude_m": dist([e["defender_position"][2] for e in episodes]),
        "defender_initial_distance_to_attacker_m": dist([e["defender_initial_distance_m"] for e in episodes]),
        "placement_attempts": {"max": int(max(e["placement_attempts"] for e in episodes)), "resampled_fraction": float(np.mean([e["placement_attempts"] > 1 for e in episodes]))},
        "a0_interception_fraction": float(intercepted.mean()), "a0_capture_fraction": float((outcome == "soldier_caught").mean()),
        "a0_other_fraction": float(1.0 - intercepted.mean() - (outcome == "soldier_caught").mean()), "a0_outcomes": {k: int((outcome == k).sum()) for k in sorted(set(outcome.tolist()))},
        "defender_visible_at_start_fraction": float(np.mean([e["defender_visible_initially"] for e in episodes])),
        "a0_defender_visible_episode_fraction": float(np.mean([e["defender_visible_steps"] > 0 for e in episodes])),
        "a0_gate_entry_to_interception_s": dist(lead) if lead else None, "a0_gate_entry_to_interception_n": len(lead),
        "a0_min_defender_separation_m": dist([e["min_defender_separation_m"] for e in episodes]),
    }


def protocol_specification(parent: dict | None = None) -> dict:
    return {
        "hypothesis": "After expert initialization establishes reliable goal-directed pursuit, PPO can learn evasive deviations around a stationary interception threat without destroying the "
                      "learned pursuit skill.",
        "tests": "pursuit preservation + elementary adversarial avoidance",
        "identifiers": {"training_protocol": TRAINING_PROTOCOL, "placement_protocol": PLACEMENT_PROTOCOL, "action": ACTION_VERSION, "observation": OBSERVATION_VERSION, "reward": REWARD_VERSION,
                        "curriculum": CURRICULUM, "stage": STAGE},
        "environment": {"clutter": "C0 (no static or dynamic obstacles)", "config": "identical to the Stage-1-v1 configuration (canonical C0 V2 + passive-harness defender w/o standby sync + v_s = 0)",
                        "env_config_sha256": config_sha256(), "differences_from_canonical": config_differences_from_canonical(),
                        "asset": "stationary (v_s = 0), canonical origin", "initial_geometry": "Stage-1 distribution: perimeter spawn, altitude U(10,30), initial heading offset U(0,90) deg off the line of sight",
                        "defender": "stationary, zero velocity, never acts; ACTIVE under canonical swept interception (intercept_radius 2.5 m unchanged); interception terminates the episode",
                        "defender_observation": "canonical gate: flag/pos/vel features nonzero only when the TRUE 3-D attacker-defender range <= enemy_defender_detection_range (25 m); zeros otherwise"},
        "placement": {"protocol": PLACEMENT_PROTOCOL, "lambda": list(LAMBDA_RANGE), "perpendicular_offset_m": [0.0, OFFSET_MAX_M], "orientation": "uniform angle about the local path tangent",
                      "reference": "noise-free canonical-A0 nominal path from the Stage-2 initial state (lambda along its arc length); amendment of the original initial-chord reference, approved by the "
                                   "user after the chord design gave A0 interception 51.5% < 70% (scenario_validation_initial_STOP.json, no training had started)",
                      "validity": "|x|,|y| <= L and 0 <= z <= max_altitude; invalid draws re-sampled (never clipped), <= %d attempts" % MAX_PLACEMENT_ATTEMPTS,
                      "stream": f"SeedSequence([{hex(PLACEMENT_TAG)}, scenario_seed])", "stop_rule": f"A0 interception fraction < {A0_INTERCEPTION_MIN} on {PLACEMENT_VALIDATION} => STOP, do not tune after training starts"},
        "initialization": {"actor": "matching Stage-1 BC checkpoint (mlp_extractor.policy_net + action_net + log_std = -2.0)", "critic": "FRESH, initialized by the PPO seed (value network of PPO(seed=seed))",
                           "optimizer": "fresh (no BC optimizer state)", "not_loaded": ["failed A1-v1 PPO weights", "Stage-1-v1 failed PPO models"], "parents": parent or {}},
        "reward": REWARD_VERSION + " unchanged (no Stage-2 reward terms)",
        "ppo": {"policy": A.A1_V2_POLICY, "hyperparameters": A.A1_V2_PPO_HYPERPARAMS, "n_envs": N_ENVS, "rollout_steps": N_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"],
                "max_rollouts": MAX_ROLLOUTS, "max_timesteps": MAX_ROLLOUTS * N_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"], "seeds": list(PPO_SEEDS)},
        "validation": {"A": list(VALIDATION_A), "B": list(VALIDATION_B), "alternation": "rollout k: odd -> A, even -> B", "anchor": list(ANCHOR),
                       "final": "147000-147199 (200 episodes per seed) + matched canonical A0", "namespace_overlaps": namespace_overlaps()},
        "gate": GATE, "mastery": "two consecutive rollout validations pass ALL gate checks",
        "retention": f"anchor capture < {ANCHOR_RETENTION_MIN} on {ANCHOR_FAILURE_STREAK} consecutive rollouts => PURSUIT RETENTION FAILURE (seed stops)",
        "success_rule": ">= 2/3 seeds master, mastered seeds retain Stage-1 pursuit, no shared pathology",
        "paired_statistics": {"bootstrap": "paired percentile bootstrap over scenarios", "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }


def protocol_sha256(parent: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(protocol_specification(parent), sort_keys=True, default=str).encode()).hexdigest()
