"""Curriculum Stage 4A-1 (`a1_curriculum_stage4a1`) for the V2 learned attacker A1: teach GOAL-CONDITIONED pursuit of a STATIONARY asset at a RANDOMIZED (translated) position, by supervised imitation of canonical A0.

Why: Stage 4A (`journal_v2_a1_cur4a.py`) showed that the Stage3-v2 parents barely use the asset-position observation slots (first-layer weight columns at the untrained orthogonal-init value; direction
sensitivity to the asset position ~2% of the sensitivity to the attacker's own position) because the asset sat at the origin in every one of Stages 1-3. Stage 4A-1 is a representation/curriculum
intervention, not a new controller: it keeps the EXACT Stage1-v2 no-defender-interaction mechanism (`journal_v2_a1_cur1v2.Stage1V2Env`: defender parked, observation masked, interception impossible,
canonical perimeter spawn) and the EXACT policy architecture (37-D in, MLP [64,64] tanh, 3-D Gaussian actor, `log_std` frozen at -2.0), but replaces the fixed-origin asset with a RANDOMLY TRANSLATED one:
both the attacker's and the asset's initial positions are shifted by the SAME horizontal vector delta, so the relative pursuit geometry (and the endurance budget, which depends only on the initial
3-D distance) is UNCHANGED, while the absolute (and hence observed) asset/attacker positions vary across a wide region of the arena. The asset still does not move during an episode (`v_s = 0`); only
its initial position varies episode-to-episode. Training is supervised imitation of canonical A0 (`MSE + (1-cos)` on the normalized expert direction), combined with a small behaviour-retention term
that keeps the fine-tuned actor close to its own Stage3-v2 parent's actions on reserved Stage3-v2 (moving-defender) scenarios, so Stage-3 evasion is not overwritten. NO PPO `learn()` is ever called here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur4 as P4
from experiments import journal_v2_a1_cur4a as P4A
from experiments import journal_v2_a1_r2 as R

VERSION = "a1_curriculum_stage4a1"
CURRICULUM = "journal_v2_a1_curriculum"
STAGE = "stage4a1"
ACTION_VERSION, OBSERVATION_VERSION, REWARD_VERSION = V1V2.ACTION_VERSION, V1V2.OBSERVATION_VERSION, V1V2.REWARD_VERSION
BC_PROTOCOL = "a1_cur4a1_goal_conditioned_bc_v1"
CURRICULUM_MODE = "curriculum_translated_goal_conditioned_imitation"
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE
DOCS = "docs/journal_v2_a1_curriculum/stage4a1/"
STAGE4A_COMMIT = "9984495"
PARENT_STATUS, PARENT_SHA256 = P4.PARENT_STATUS, P4.PARENT_SHA256
STAGE4_ACCEPTED_FINAL_SHA256, STAGE4A_ACCEPTED_FINAL_SHA256 = P4A.STAGE4_ACCEPTED_FINAL_SHA256, {
    201: "4b94fe90d05ecdfa8f894894447251b7c102f3a840a07b249e9ffdac304dde95", 202: "26ed0b49b462b52ddc62509de7eca5f25441a4e1d7f1b4f5a9b488899904e343",
    203: "bf2f5d97a90b6959e8e551f252a9c5a6f7dc92548d06a3a0492bc19a2bc77938"}      # failed Stage-4 / Stage-4A final checkpoints (non-canonical evidence; never used for initialisation)
parent_checkpoint = P4.parent_checkpoint
PPO_SEEDS = X.PPO_SEEDS

# ----------------------------------------------------------------------------- pre-registered constants and namespaces (frozen before any data is generated)
TRANSLATION_STREAM_TAG = 0xC4A2        # dedicated translation-delta stream, independent of spawn/soldier/enemy/sensor/obstacle streams
MARGIN_M = 5.0                          # 10% of L = 50 m: minimum clearance of either translated position from the arena wall
EXPERT_SEED_START, EXPERT_EPISODES = 172_000, 6_000           # translated A0 expert episodes (172000-177999)
EXPERT_SHARDS = 12
TRAIN_FRACTION = 0.8
VALIDATION_A = (178_000, 178_099)                             # held-out translated scenarios (unseen episodes, same full-coverage translation distribution)
VALIDATION_B = (178_100, 178_199)
STAGE1_ANCHOR = (178_200, 178_299)      # Anchor 1: Stage-1-v2 fixed-ORIGIN pursuit (unchanged task; stricter 99% gate at this stage)
STATIC_ANCHOR = (178_300, 178_399)      # Anchor 2: Stage-2 stationary defender, stationary (origin) asset
MOVING_DEFENDER_ANCHOR = (178_400, 178_499)   # Anchor 3: Stage3-v2 hold/release Greedy, stationary (origin) asset
GENERALIZATION = (178_500, 179_499)      # translation-GENERALIZATION held-out: stratified quadrant x magnitude, translations NOT used in BC/held-out (shared scan pool; ~35-45% feasibility per stratum)
RETENTION_SOURCE = (191_200, 192_199)    # scenarios used to build the per-seed Stage3-v2 behaviour-retention (rehearsal) dataset
N_ENVS, PPO_SEEDS_ = X.N_ENVS, PPO_SEEDS
MAX_EPOCHS = 20
RETENTION_TARGET_PER_SEED = 12_000
RETAIN_LOSS_WEIGHT = 0.25                # frozen before training: L = L_goal + 0.25 * L_retain
ANCHOR_MIN = {"anchor1_basic_pursuit": 0.99, "anchor2_static_defender": 0.80, "anchor3_moving_defender": 0.75}
ANCHOR_FAILURE_STREAK = 2
SENSITIVITY_RATIO_MIN_GAIN = 2.0         # POST asset/attacker sensitivity ratio must be >= 2x PRE for a passing seed
QUADRANT_CAPTURE_MIN = 0.95              # no translation-generalization region may fall below this for a passing seed

GATE = {
    "capture_min": 0.99, "exhaustion_max": 0.0, "timeout_max": 0.0, "median_path_excess_max": 1.20, "gt2_max": 0.01, "gt5_max": 0.0, "orbit_episodes_max": 0,
    "cosine_mean_min": 0.995, "cosine_min_min": 0.95, "anchor1_capture_min": ANCHOR_MIN["anchor1_basic_pursuit"], "anchor2_capture_min": ANCHOR_MIN["anchor2_static_defender"],
    "anchor3_capture_min": ANCHOR_MIN["anchor3_moving_defender"], "nonfinite_max": 0,
}
RESERVED_RANGES = {**P4A.RESERVED_RANGES, "stage4a_validation_A": P4A.VALIDATION_A, "stage4a_validation_B": P4A.VALIDATION_B, "stage4a_anchor1": P4A.STAGE1_ANCHOR, "stage4a_anchor2": P4A.STATIC_ANCHOR,
                   "stage4a_anchor3": P4A.MOVING_DEFENDER_ANCHOR, "stage4a_baseline": P4A.BASELINE}


def seeds_of(rng_tuple) -> range:
    return range(rng_tuple[0], rng_tuple[1] + 1)


def expert_seeds() -> range:
    return range(EXPERT_SEED_START, EXPERT_SEED_START + EXPERT_EPISODES)


def validation_seeds(name: str) -> range:
    return seeds_of({"A": VALIDATION_A, "B": VALIDATION_B}[name])


def final_evaluation_seeds() -> list[int]:
    return list(validation_seeds("A")) + list(validation_seeds("B"))


def namespace_overlaps() -> dict:
    own = {"expert": (EXPERT_SEED_START, EXPERT_SEED_START + EXPERT_EPISODES - 1), "validation_A": VALIDATION_A, "validation_B": VALIDATION_B, "anchor1": STAGE1_ANCHOR, "anchor2": STATIC_ANCHOR,
           "anchor3": MOVING_DEFENDER_ANCHOR, "generalization": GENERALIZATION, "retention_source": RETENTION_SOURCE}
    out = {name: [k for k, (a, b) in RESERVED_RANGES.items() if not (hi < a or lo > b)] for name, (lo, hi) in own.items()}
    names = list(own)
    out["mutual"] = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if not (own[a][1] < own[b][0] or own[a][0] > own[b][1])]
    out["training_stream"] = [C.TRAIN_SEED_LOW, C.TRAIN_SEED_HIGH]
    out["training_overlap"] = [n for n, (lo, hi) in own.items() if not (hi < C.TRAIN_SEED_LOW or lo >= C.TRAIN_SEED_HIGH)]
    return out


# ----------------------------------------------------------------------------- configuration (unchanged Stage-1-v2 config; asset still stationary, v_s = 0)
stage4a1_config = V1V2.stage1v2_config
config_sha256 = V1V2.config_sha256
config_differences_from_canonical = V1V2.config_differences_from_canonical
DEFENDER_SLOTS = P4A.DEFENDER_SLOTS
ASSET_SLOTS, ATTACKER_SLOTS = (3, 4, 5), (0, 1, 2)


class TranslationInfeasible(Exception):
    """Raised when a requested (stratified) translation does not fit the per-scenario feasible box; callers retry with a different scenario seed."""


def feasible_box(enemy0_xy: np.ndarray, L: float, margin: float) -> np.ndarray:
    """[[lo_x, hi_x], [lo_y, hi_y]]: translations for which BOTH the (origin-starting) asset and the edge-spawned attacker stay within [-(L-margin), L-margin]^2. Exact (no rejection needed): the soldier
    always starts at the origin, so the soldier constraint is |delta_i| <= L-margin and the attacker constraint is |enemy0_i + delta_i| <= L-margin; the intersection is always non-empty for margin < L/2."""
    lim = float(L) - float(margin)
    out = np.empty((2, 2), dtype=np.float64)
    for i in range(2):
        e = float(enemy0_xy[i])
        lo = max(-lim, -lim - e)
        hi = min(lim, lim - e)
        out[i] = (lo, hi)
    return out


def sample_translation(scenario_seed: int, enemy0_xy: np.ndarray, L: float, margin: float, *, target: tuple[float, float] | None = None, jitter_m: float = 0.0) -> np.ndarray:
    """Uniform sample inside the feasible box (`target=None`, the training/held-out mode), or a validity-checked point near `target` (+/- `jitter_m`, the stratified-generalization mode). Raises
    `TranslationInfeasible` if `target` does not fit; callers then retry with a different scenario seed (never silently clipped)."""
    box = feasible_box(enemy0_xy, L, margin)
    rng = np.random.default_rng(np.random.SeedSequence([TRANSLATION_STREAM_TAG, int(scenario_seed)]))
    if target is None:
        return np.array([rng.uniform(box[0, 0], box[0, 1]), rng.uniform(box[1, 0], box[1, 1])])
    tx, ty = target
    lo_x, hi_x = max(box[0, 0], tx - jitter_m), min(box[0, 1], tx + jitter_m)
    lo_y, hi_y = max(box[1, 0], ty - jitter_m), min(box[1, 1], ty + jitter_m)
    if lo_x > hi_x or lo_y > hi_y:
        raise TranslationInfeasible(f"target {target} +/- {jitter_m} m does not fit the feasible box {box.tolist()} for enemy0={enemy0_xy.tolist()}")
    return np.array([rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)])


class TranslatedStage1V2Env(V1V2.Stage1V2Env):
    """Stage-1-v2 clean curriculum environment (masked parked defender, no interception, C0) with BOTH the attacker and the (still-stationary) asset translated by the SAME horizontal vector, so the
    relative pursuit geometry and endurance budget are preserved while the absolute positions vary. `options={"scenario": seed, "target_translation": (tx, ty), "translation_jitter_m": j}` selects the
    stratified mode; omitting `target_translation` samples uniformly from the per-scenario feasible box."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        env = self._env
        L, margin = float(self._config.L), MARGIN_M
        enemy0_xy = np.asarray(env._enemy_pos[:2], dtype=np.float64)
        target = (options or {}).get("target_translation")
        jitter = float((options or {}).get("translation_jitter_m", 0.0))
        delta = sample_translation(self.scenario_seed, enemy0_xy, L, margin, target=target, jitter_m=jitter)
        env._soldier_pos = (env._soldier_pos + np.array([delta[0], delta[1], 0.0], dtype=np.float32)).astype(np.float32)
        env._enemy_pos = (env._enemy_pos + np.array([delta[0], delta[1], 0.0], dtype=np.float32)).astype(np.float32)
        box = feasible_box(enemy0_xy, L, margin)
        assert (box[0, 0] - 1e-6 <= delta[0] <= box[0, 1] + 1e-6) and (box[1, 0] - 1e-6 <= delta[1] <= box[1, 1] + 1e-6), "translation outside its own feasible box"
        assert np.abs(env._soldier_pos[:2]).max() <= L - margin + 1e-6 and np.abs(env._enemy_pos[:2]).max() <= L - margin + 1e-6, "translated position violates the safety margin"
        self.translation = delta
        self._asset0 = np.asarray(env._soldier_pos, dtype=np.float64).copy()
        self._start_pos = np.asarray(env._enemy_pos, dtype=np.float64).copy()
        self._prev_pos = self._start_pos.copy()
        self._range = self._range_now()
        self._initial_range = self._range
        return self._observation(), self._info({}, None)

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        out = super()._episode_summary(outcome, path_now, range_next)
        out.update(translation_m=[float(self.translation[0]), float(self.translation[1])], asset_position_m=[float(x) for x in self._asset0])
        return out


def make_env(ppo_seed: int = 0, worker_index: int = 0, rollout_index: int = 0, **kwargs) -> TranslatedStage1V2Env:
    return TranslatedStage1V2Env(ppo_seed, worker_index, rollout_index, **kwargs)


# ----------------------------------------------------------------------------- stratified translation-generalization regions (held out; never used for BC/held-out validation)
def generalization_strata() -> dict:
    """8 strata: 4 diagonal quadrant directions x {inner, outer} magnitude, chosen to be far from both the origin (already exercised by Stages 1-3) and the BULK of the full-coverage BC/held-out
    distribution (whose per-scenario box is edge-dependent and therefore NOT uniform over this exact grid)."""
    inner_r, outer_r = 15.0, 35.0
    out = {}
    for qx, qy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for band, r in (("inner", inner_r), ("outer", outer_r)):
            name = f"{'pos' if qx > 0 else 'neg'}x_{'pos' if qy > 0 else 'neg'}y_{band}"
            out[name] = (qx * r / np.sqrt(2.0), qy * r / np.sqrt(2.0))
    return out


def stratified_scenarios(stratum_name: str, target: tuple[float, float], n: int, *, jitter_m: float = 3.0, max_scan: int = 2_000) -> list[int]:
    """`n` scenario seeds (shared scan pool `GENERALIZATION`, same starting point for every stratum) for which `target` (+/- jitter) fits the per-scenario feasible box; infeasible candidates are
    skipped (never translation-clipped). A seed may be accepted by more than one stratum (different target translations => genuinely different episodes, not duplicate evaluations)."""
    accepted: list[int] = []
    env = TranslatedStage1V2Env(0, 0, 0, scripted_a0=True)
    scanned = 0
    for candidate in range(GENERALIZATION[0], GENERALIZATION[1] + 1):
        if len(accepted) >= n or scanned >= max_scan:
            break
        scanned += 1
        env2 = V1V2.Stage1V2Env(0, 0, 0, scripted_a0=True)      # cheap untranslated reset just to read the edge spawn / feasible box
        env2.reset(options={"scenario": candidate})
        enemy0_xy = np.asarray(env2.soldier_env._enemy_pos[:2], dtype=np.float64)
        box = feasible_box(enemy0_xy, float(env2._config.L), MARGIN_M)
        tx, ty = target
        if box[0, 0] - jitter_m <= tx <= box[0, 1] + jitter_m and box[1, 0] - jitter_m <= ty <= box[1, 1] + jitter_m:
            try:
                env.reset(options={"scenario": candidate, "target_translation": target, "translation_jitter_m": jitter_m})
                accepted.append(candidate)
            except TranslationInfeasible:
                pass
    return accepted


def generalization_scenario_table(per_stratum: int = 40) -> dict:
    return {name: stratified_scenarios(name, target, per_stratum) for name, target in generalization_strata().items()}


# ----------------------------------------------------------------------------- evaluation / metrics / gate (reuses the Stage-1-v2 expert-cosine machinery)
evaluate = V1V2.evaluate
metrics = V1V2.metrics
evaluate_anchor1 = lambda act, seeds=None: V1V2.evaluate(act, list(seeds if seeds is not None else seeds_of(STAGE1_ANCHOR)))          # noqa: E731
evaluate_anchor2 = lambda act, seeds=None: S.evaluate(act, list(seeds if seeds is not None else seeds_of(STATIC_ANCHOR)))              # noqa: E731
evaluate_anchor3 = lambda act, seeds=None: X.evaluate(act, list(seeds if seeds is not None else seeds_of(MOVING_DEFENDER_ANCHOR)))      # noqa: E731
anchor_metrics_static, anchor_metrics_moving = P4A.anchor_metrics, P4A.anchor_metrics


def evaluate_translated(act, seeds, *, scripted_a0: bool = False, target: tuple[float, float] | None = None, jitter_m: float = 0.0) -> list[dict]:
    env = TranslatedStage1V2Env(0, 0, 0, scripted_a0=scripted_a0)
    episodes = []
    for s in seeds:
        obs, _ = env.reset(options={"scenario": int(s), **({"target_translation": target, "translation_jitter_m": jitter_m} if target is not None else {})})
        done, info, cosines = False, {}, []
        while not done:
            action = np.zeros(3, dtype=np.float32) if scripted_a0 else np.asarray(act(obs), dtype=np.float32)
            if not scripted_a0:
                u, _held = C.desired_direction(action, env.soldier_env._enemy_vel)
                cosines.append(float(np.dot(u, env.expert_action())))
            obs, _r, te, tr, info = env.step(action)
            done = te or tr
        episode = dict(info["a1_episode"])
        episode["expert_cosines"] = cosines
        episodes.append(episode)
    return episodes


def translated_metrics(episodes: list[dict]) -> dict:
    out = V1V2.metrics(episodes)
    return out


def gate_status(m: dict, *, anchor1: float, anchor2: float, anchor3: float) -> dict:
    checks = {
        "capture": m["capture"] >= GATE["capture_min"], "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"], "timeout": m["timeout"] <= GATE["timeout_max"],
        "median_path_excess": m["median_path_excess"] <= GATE["median_path_excess_max"], "gt2": m["gt2"] <= GATE["gt2_max"], "gt5": m["gt5"] <= GATE["gt5_max"],
        "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"], "cosine_mean": (m["expert_cosine_mean"] or 0.0) >= GATE["cosine_mean_min"],
        "cosine_min": (m["expert_cosine_min"] if m["expert_cosine_min"] is not None else -1.0) >= GATE["cosine_min_min"],
        "anchor1_basic_pursuit": anchor1 >= GATE["anchor1_capture_min"], "anchor2_static_defender": anchor2 >= GATE["anchor2_capture_min"], "anchor3_moving_defender": anchor3 >= GATE["anchor3_capture_min"],
        "numerical_failures": m["nonfinite"] <= GATE["nonfinite_max"],
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


# ----------------------------------------------------------------------------- representation sensitivity (shared with Stage 4A's diagnostic; reused verbatim)
def collect_observations(act, seeds, max_episodes: int = 40, *, env_factory=None) -> np.ndarray:
    env = (env_factory or (lambda: TranslatedStage1V2Env(0, 0, 0)))()
    rows = []
    for s in list(seeds)[:max_episodes]:
        obs, _ = env.reset(options={"scenario": int(s)})
        done = False
        while not done:
            rows.append(obs.copy())
            obs, _r, te, tr, _i = env.step(np.asarray(act(obs), dtype=np.float32))
            done = te or tr
    return np.array(rows, dtype=np.float32)


def asset_input_sensitivity(model, obs: np.ndarray) -> dict:
    """Same diagnostic as `journal_v2_a1_cur4a_final.asset_input_analysis_of`: first-layer weight-column norms of the asset vs attacker position slots, and the Jacobian of the executed unit direction
    w.r.t. the asset xy slots vs the attacker xy slots (ratio ~100% with opposite sign = perfect relative tracker; ~0% = an absolute-coordinate aimer)."""
    import torch

    w = model.policy.mlp_extractor.policy_net[0].weight.detach().numpy()
    x = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
    mean = model.policy.get_distribution(x).distribution.mean
    unit = mean / mean.norm(dim=1, keepdim=True).clamp_min(1e-8)
    jac = np.stack([torch.autograd.grad(unit[:, k].sum(), x, retain_graph=True)[0].numpy() for k in range(3)], axis=1)
    a, b = jac[:, :, 3:5].reshape(len(obs), -1), jac[:, :, 0:2].reshape(len(obs), -1)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    align = -np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-12)
    return {"first_layer_column_norm_asset_xyz": [float(v) for v in np.linalg.norm(w[:, list(ASSET_SLOTS)], axis=0)],
            "first_layer_column_norm_attacker_xyz": [float(v) for v in np.linalg.norm(w[:, list(ATTACKER_SLOTS)], axis=0)], "orthogonal_init_column_norm": float(np.sqrt(2.0)),
            "jacobian_norm_asset_xy_mean": float(na.mean()), "jacobian_norm_attacker_xy_mean": float(nb.mean()), "asset_to_attacker_sensitivity_ratio": float(na.mean() / nb.mean()),
            "relative_alignment_mean": float(align.mean()), "states": int(len(obs))}


def representation_record(model, *, held_out_seeds=None, per_stratum: int = 40) -> dict:
    """PRE/POST-comparable snapshot: asset-input sensitivity (on the translated held-out set), translated held-out metrics, and capture/cosine by translation-generalization region."""
    from experiments import journal_v2_a1_cur1 as C_

    act = C_.model_actor(model)
    seeds = list(held_out_seeds if held_out_seeds is not None else final_evaluation_seeds())
    obs = collect_observations(act, seeds)
    sens = asset_input_sensitivity(model, obs)
    held = translated_metrics(evaluate_translated(act, seeds))
    strata = generalization_strata()
    table = generalization_scenario_table(per_stratum=per_stratum)
    regions = {}
    for name, target in strata.items():
        eps = []
        for s in table[name]:
            eps += evaluate_translated(act, [s], target=target, jitter_m=3.0)
        regions[name] = translated_metrics(eps) if eps else {"capture": None, "episodes": 0}
    return {"sensitivity": sens, "held_out": {k: held[k] for k in ("capture", "exhaustion", "timeout", "median_path_excess", "gt2", "gt5", "orbit_episodes", "nonfinite", "expert_cosine_mean",
                                                                   "expert_cosine_median", "expert_cosine_p5", "expert_cosine_min")},
            "generalization_regions": {name: {k: regions[name].get(k) for k in ("capture", "expert_cosine_mean", "episodes")} for name in regions},
            "generalization_scenarios": table}


# ----------------------------------------------------------------------------- protocol
def protocol_specification() -> dict:
    v4a = P4A.protocol_specification()
    return {
        "version": VERSION, "curriculum_mode": CURRICULUM_MODE,
        "hypothesis": "The Stage-4 moving-target failure was caused in part by inadequate dependence on asset-position inputs created by fixed-origin curriculum training. Randomizing the stationary "
                      "goal location during expert imitation should produce translation-aware pursuit while preserving previously learned defender-evasion behavior.",
        "tests": "representation/curriculum intervention: randomized-origin goal conditioning by A0 imitation (NOT a new controller architecture, NOT PPO learn())",
        "identifiers": {"bc_protocol": BC_PROTOCOL, "action": ACTION_VERSION, "observation": OBSERVATION_VERSION, "reward": REWARD_VERSION, "curriculum": CURRICULUM, "stage": STAGE},
        "prior_evidence": {"stage4": {"commit": "15faece", "result": "0/3 mastered; repeated fly-bys"},
                           "stage4a": {"commit": STAGE4A_COMMIT, "result": "0/3 mastered; asset-input sensitivity ratio only 2.1-2.5% PRE (Stage3-v2 parents), ~10% POST Stage-4A (393k PPO steps)",
                                      "final_sha256": STAGE4A_ACCEPTED_FINAL_SHA256}},
        "parents": {"status": PARENT_STATUS, "checkpoints": P4.parent_hashes(), "note": "accepted Stage3-v2 curriculum checkpoints (NOT the failed Stage-4/4A models)"},
        "environment": {"mechanism": "unchanged Stage-1-v2 curriculum_basic_pursuit_no_defender_interaction (defender parked + masked + non-intercepting)", "asset": "STATIONARY within an episode (v_s = 0); "
                        "position is TRANSLATED (not moved during the episode)", "env_config_sha256": config_sha256(), "config_differences_from_canonical": config_differences_from_canonical(),
                        "translation": {"mechanism": "both the attacker's canonical perimeter-spawn position and the asset's origin are shifted by the SAME horizontal vector delta, drawn from a "
                                       f"dedicated stream (tag {hex(TRANSLATION_STREAM_TAG)}) independent of spawn/soldier/enemy/sensor/obstacle streams; relative geometry and the endurance budget "
                                       "(computed from the initial 3-D distance) are therefore exactly preserved", "margin_m": MARGIN_M,
                                       "feasible_box": "exact per-scenario box from the edge-spawn point and the origin-starting asset; no clipping, invalid targets are rejected (`TranslationInfeasible`) "
                                                       "and a different scenario seed is tried"}},
        "observation_audit": "a1_v2_obs_v1 (37-D) unchanged; no new features; translation tests whether the EXISTING asset-position slots (3-5) can be learned to carry goal information.",
        "architecture": "unchanged SB3 MlpPolicy [64,64] tanh, 37-D -> 3-D Gaussian; ONLY mlp_extractor.policy_net and action_net are trained (actor only); log_std frozen at -2.0; critic untouched.",
        "expert": "canonical scripted A0 (direct_obstacle_only): normalized desired direction unit(p_asset - p_attacker) at the CURRENT (translated) state; no privileged/future information.",
        "dataset": {"expert_scenarios": [EXPERT_SEED_START, EXPERT_SEED_START + EXPERT_EPISODES - 1], "episodes": EXPERT_EPISODES, "target_pairs": "60,000-80,000",
                    "split": "by episode, 80% train / 20% val (seeded permutation, Stage-1-v2 split mechanism)"},
        "retention_dataset": {"source": "reserved Stage3-v2 (moving-defender) scenarios, driven by the SEED'S OWN Stage3-v2 parent policy (not A0)", "seeds": list(RETENTION_SOURCE),
                              "target_states_per_seed": RETENTION_TARGET_PER_SEED, "label": "parent policy deterministic mean action at the visited state",
                              "state_classes": ["not_visible", "newly_visible", "active_evasion", "post_pass_pursuit"], "no_future_outcome_labels": True},
        "objective": {"L_goal": "MSE(a, a_A0) + (1 - cos(a, a_A0))", "L_retain": "MSE(a, a_parent) + (1 - cos(a, a_parent))", "L_total": f"L_goal + {RETAIN_LOSS_WEIGHT} * L_retain",
                      "weight_frozen_before_training": RETAIN_LOSS_WEIGHT, "trained_parameters": ["mlp_extractor.policy_net", "action_net"], "log_std": -2.0, "max_epochs": MAX_EPOCHS,
                      "epoch_selection": "lowest combined validation loss", "no_ppo_learn": True},
        "validation": {"A": list(VALIDATION_A), "B": list(VALIDATION_B), "final": "178000-178199 (200 episodes), translated full-coverage held-out", "anchor1_basic_pursuit": list(STAGE1_ANCHOR),
                       "anchor2_static_defender": list(STATIC_ANCHOR), "anchor3_moving_defender": list(MOVING_DEFENDER_ANCHOR), "generalization": list(GENERALIZATION),
                       "retention_source": list(RETENTION_SOURCE), "namespace_overlaps": namespace_overlaps()},
        "gate": GATE, "mastery": "two consecutive epoch validations pass ALL gate checks",
        "retention": {"anchors": ANCHOR_MIN, "stop": f"capture below the anchor minimum on {ANCHOR_FAILURE_STREAK} consecutive epochs on ANY anchor => retention failure"},
        "representation_gate": {"asset_sensitivity_ratio_min_gain": SENSITIVITY_RATIO_MIN_GAIN, "quadrant_capture_min": QUADRANT_CAPTURE_MIN,
                                "rule": "for each PASSING seed: POST asset/attacker sensitivity ratio >= 2x PRE, AND no translation-generalization region captures below 95%"},
        "success_rule": ">= 2/3 seeds master, representation gate passes, Stage1/2/3 retention holds, no shared pathology; < 2 => STOP, do not proceed to moving-asset imitation",
        "generalization_strata": {k: list(v) for k, v in generalization_strata().items()},
        "prior_stage4a_protocol_sha256": v4a and hashlib.sha256(json.dumps(v4a, sort_keys=True, default=str).encode()).hexdigest(),
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()
