"""Curriculum Stage 1 V2 — EXPERT INITIALIZATION of A1 basic pursuit by behavior cloning from canonical A0. NO PPO learn() anywhere in this stage.

Mode `curriculum_basic_pursuit_no_defender_interaction` (a CURRICULUM-ONLY configuration/wrapper; canonical environment behavior is untouched):
  * C0, stationary asset (`v_s = 0`), canonical attacker dynamics, finite endurance k = 5, canonical swept asset capture, Stage-1-v1 randomized geometry (canonical
    perimeter spawn + random initial heading offset <= 90 deg);
  * the defender is structurally present and STATIONARY (parked at a ground corner, never acts) but (a) its tactical observation is MASKED as unavailable (the attacker
    observation builder is called with a negative detection range, so the defender-available flag and defender position/velocity slots are exactly 0 in every state), and
    (b) it cannot terminate an episode: `intercept_radius` is set to 1e-6 m (config only), so interception is geometrically impossible.
  * observation `a1_v2_obs_v1` (37-D, all defender and obstacle slots kept), action `a1_cur1_action_v1` (continuous unit direction; no A0 fallback at execution time),
    reward `a1_v2_reward_v2` (unchanged; unused by behavior cloning).

Expert label = the normalized A0 desired-direction command for the SAME attacker/asset state: in C0 A0 flies `u = (p_asset - p_attacker)/|.|` (`direct_obstacle_only`,
no obstacles, no route planning, evasion zeroed). Computed from the current state only (no future asset information, no privileged trajectory labels).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_r2 as R

CURRICULUM_MODE = "curriculum_basic_pursuit_no_defender_interaction"
STAGE = "stage1_v2"
ACTION_VERSION = C.ACTION_VERSION                       # a1_cur1_action_v1
OBSERVATION_VERSION = A.A1_V2_OBSERVATION_VERSION       # a1_v2_obs_v1
REWARD_VERSION = R.REWARD_VERSION                       # a1_v2_reward_v2
BC_PROTOCOL = "a1_cur1v2_bc_v1"
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE
MASKED_DETECTION_RANGE = -1.0                           # dist <= -1 never holds -> defender features unavailable
NO_INTERCEPT_RADIUS = 1e-6
PPO_SEEDS = C.PPO_SEEDS                                 # 201, 202, 203 (independent BC initializations)

# ----------------------------------------------------------------------------- pre-registered namespaces and hyper-parameters (frozen before any data is generated)
EXPERT_SEED_START, EXPERT_EPISODES, EXPERT_SHARDS = 160_000, 5_000, 10          # expert scenarios 160000-164999 (500 per shard)
RELABEL_SEED_START, RELABEL_EPISODES, RELABEL_MAX_STEPS = 170_000, 1_000, 100   # relabeling scenarios 170000-170999, <= 100 visited states per episode
TRAIN_FRACTION = 0.8
SPLIT_TAG = 0xC1B1
BC = {"lr": 1e-3, "batch_size": 256, "epochs": 100, "loss": "mse + (1 - cosine)", "optimizer": "Adam", "model_selection": "lowest validation loss over epochs"}
FINE_TUNE = {"lr": 5e-4, "batch_size": 256, "epochs": 50, "loss": "mse + (1 - cosine)", "optimizer": "Adam", "model_selection": "lowest validation loss over epochs",
             "start": "round-0 selected weights", "rounds": 1}
LOG_STD_INIT = -2.0
MASK_TRAINED_PARAMETERS = ("mlp_extractor.policy_net", "action_net")             # actor only; value network and log_std are not trained by BC
RESERVED_RANGES = {"stage1_v1_validation_and_final": (146_000, 146_199), "v2_a0_validation": (130_000, 130_099), "a1_eval": (140_000, 140_099), "diagnostic": (145_000, 145_099),
                   "a1_validation": (150_000, 150_039), "v1_a1_validation": (180_000, 180_999), "v1_a1_heldout": (190_000, 190_999), "v1_clutter_eval": (95_000, 95_999),
                   "stage1_v1_training_stream": (C.TRAIN_SEED_LOW, C.TRAIN_SEED_HIGH)}

GATE_V2 = {"capture_min": 0.98, "exhaustion_max": 0.01, "timeout_max": 0.01, "median_path_excess_max": 1.20, "gt2_max": 0.02, "gt5_max": 0.0, "orbit_episodes_max": 0, "nonfinite_max": 0}
COMMON_PATHOLOGY_RATE, COMMON_PATHOLOGY_SEEDS = 0.01, 2


def expert_seeds() -> range:
    return range(EXPERT_SEED_START, EXPERT_SEED_START + EXPERT_EPISODES)


def relabel_seeds() -> range:
    return range(RELABEL_SEED_START, RELABEL_SEED_START + RELABEL_EPISODES)


def namespace_overlaps() -> dict:
    """Overlap of the expert / relabel scenario ranges with every reserved range (must be empty)."""
    out = {}
    for name, ranges in (("expert", expert_seeds()), ("relabel", relabel_seeds())):
        lo, hi = ranges.start, ranges.stop - 1
        out[name] = [k for k, (a, b) in RESERVED_RANGES.items() if not (hi < a or lo > b)]
    out["expert_vs_relabel"] = not (expert_seeds().stop <= relabel_seeds().start or relabel_seeds().stop <= expert_seeds().start)
    return out


def stage1v2_config():
    """C.stage1_config() (canonical C0 V2, v_s = 0, standby off) with interception made impossible by `intercept_radius = 1e-6`."""
    return dataclasses.replace(C.stage1_config(), intercept_radius=NO_INTERCEPT_RADIUS)


def config_differences_from_canonical() -> dict:
    canonical = A.build_episode_config("C0", "greedy")
    out = {}
    for f in dataclasses.fields(canonical):
        a, b = getattr(canonical, f.name), getattr(stage1v2_config(), f.name)
        if a != b:
            out[f.name] = {"canonical": a, "stage1_v2": b}
    return out


def config_sha256() -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(stage1v2_config()), sort_keys=True, default=str).encode()).hexdigest()


class Stage1V2Env(C.Stage1Env):
    """Stage-1-v1 environment geometry with the clean curriculum mode: masked defender observation, no defender termination, expert-label access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = stage1v2_config()

    def _observation(self) -> np.ndarray:
        return A.build_a1_v2_observation(self._env, MASKED_DETECTION_RANGE).astype(np.float32)

    def expert_action(self) -> np.ndarray:
        """Unit direction A0 would command from the CURRENT attacker/asset state (what `SoldierEnv._move_enemy` computes as `r_hat` in C0)."""
        r = np.asarray(self._env._soldier_pos, dtype=np.float32) - np.asarray(self._env._enemy_pos, dtype=np.float32)
        norm = float(np.linalg.norm(r))
        if norm <= float(self._config.eps):
            raise ValueError("attacker on top of the asset: A0 would draw a random direction; such a state never occurs before termination")
        return (r / norm).astype(np.float32)


def make_env(ppo_seed: int = 0, worker_index: int = 0, rollout_index: int = 0, **kwargs) -> Stage1V2Env:
    return Stage1V2Env(ppo_seed, worker_index, rollout_index, **kwargs)


# ----------------------------------------------------------------------------- split (by EPISODE)
def split_of(scenario_seed: int, start: int, n: int, fraction: float = TRAIN_FRACTION, extra: int = 0) -> str:
    """Deterministic episode-level assignment: a seeded permutation of the episode indices; the first `fraction` are 'train', the rest 'val'."""
    index = int(scenario_seed) - int(start)
    if not 0 <= index < n:
        raise ValueError("scenario seed outside the split namespace")
    perm = np.random.default_rng(np.random.SeedSequence([SPLIT_TAG, int(start), int(n), int(extra)])).permutation(n)
    rank = int(np.where(perm == index)[0][0])
    return "train" if rank < int(round(fraction * n)) else "val"


def split_table(start: int, n: int, fraction: float = TRAIN_FRACTION, extra: int = 0) -> np.ndarray:
    """Boolean array, True = train, for every episode index in [0, n)."""
    perm = np.random.default_rng(np.random.SeedSequence([SPLIT_TAG, int(start), int(n), int(extra)])).permutation(n)
    is_train = np.zeros(n, dtype=bool)
    is_train[perm[: int(round(fraction * n))]] = True
    return is_train


# ----------------------------------------------------------------------------- evaluation (deterministic policy means) + gate
def evaluate(act, seeds, *, scripted_a0: bool = False) -> list[dict]:
    """Deterministic episodes with per-step cosine between the executed direction and the A0 label at the same state (A0 itself: cosine 1 by definition)."""
    env = Stage1V2Env(0, 0, 0, scripted_a0=scripted_a0)
    episodes = []
    for s in seeds:
        obs, _ = env.reset(options={"scenario": int(s)})
        done, info, cosines = False, {}, []
        while not done:
            if scripted_a0:
                action = np.zeros(3, dtype=np.float32)
            else:
                action = np.asarray(act(obs), dtype=np.float32)
                u, _held = C.desired_direction(action, env.soldier_env._enemy_vel)
                cosines.append(float(np.dot(u, env.expert_action())))
            obs, _r, te, tr, info = env.step(action)
            done = te or tr
        episode = dict(info["a1_episode"])
        episode["expert_cosines"] = cosines
        episodes.append(episode)
    return episodes


def metrics(episodes: list[dict]) -> dict:
    out = C.metrics(episodes)
    cos = np.array([c for e in episodes for c in e.get("expert_cosines", [])], dtype=np.float64)
    out["expert_cosine_mean"] = float(cos.mean()) if cos.size else None
    out["expert_cosine_median"] = float(np.median(cos)) if cos.size else None
    out["expert_cosine_p5"] = float(np.percentile(cos, 5)) if cos.size else None
    out["expert_cosine_min"] = float(cos.min()) if cos.size else None
    out["expert_cosine_steps"] = int(cos.size)
    return out


def gate_status(m: dict) -> dict:
    checks = {"capture": m["capture"] >= GATE_V2["capture_min"], "exhaustion": m["exhaustion"] <= GATE_V2["exhaustion_max"], "timeout": m["timeout"] <= GATE_V2["timeout_max"],
              "median_path_excess": m["median_path_excess"] <= GATE_V2["median_path_excess_max"], "gt2": m["gt2"] <= GATE_V2["gt2_max"], "gt5": m["gt5"] <= GATE_V2["gt5_max"],
              "orbit": m["orbit_episodes"] <= GATE_V2["orbit_episodes_max"], "numerical_failures": m["nonfinite"] <= GATE_V2["nonfinite_max"]}
    return {"checks": checks, "passed": bool(all(checks.values()))}


def protocol_specification() -> dict:
    return {
        "hypothesis": "Basic pursuit can be efficiently transferred from the deterministic A0 expert to A1 through imitation learning, allowing subsequent reinforcement learning to focus on "
                      "genuinely adversarial behavior rather than rediscovering elementary goal-directed navigation.",
        "mode": CURRICULUM_MODE, "versions": {"action": ACTION_VERSION, "observation": OBSERVATION_VERSION, "reward": REWARD_VERSION, "bc_protocol": BC_PROTOCOL},
        "env_config_sha256": config_sha256(), "env_config_differences_from_canonical_C0_V2": config_differences_from_canonical(),
        "stage1_v1_failed_ablation": "docs/journal_v2_a1_curriculum/stage1/ (pure PPO from scratch, 0/3 seeds mastered) is preserved and NON-CANONICAL",
        "defender_masking": {"observation": f"build_a1_v2_observation(env, detection_range={MASKED_DETECTION_RANGE}) -> defender flag/pos/vel slots exactly 0",
                             "termination": f"intercept_radius = {NO_INTERCEPT_RADIUS} m (config) -> defender cannot terminate episodes", "defender": "stationary, parked at the Stage-1-v1 ground corner, never acts, state retained"},
        "asset": "v_s = 0 (stationary); canonical reset; Stage-1-v1 randomized geometry (perimeter spawn, altitude U(10,30), initial heading offset U(0,90) deg)",
        "expert": {"teacher": "canonical scripted A0 (direct_obstacle_only), the normalized desired-direction command for the same state: unit(p_asset - p_attacker)", "privileged_information": False},
        "dataset": {"expert_scenarios": [EXPERT_SEED_START, EXPERT_SEED_START + EXPERT_EPISODES - 1], "episodes": EXPERT_EPISODES, "shards": EXPERT_SHARDS, "target_pairs": "40,000-60,000",
                    "split": f"by episode, {int(TRAIN_FRACTION * 100)}% train / {100 - int(TRAIN_FRACTION * 100)}% validation (seeded permutation, tag {hex(SPLIT_TAG)})",
                    "namespace_overlaps": namespace_overlaps()},
        "policy": {"architecture": "SB3 PPO MlpPolicy, pi/vf [64,64] tanh, 37-D input, 3-D Gaussian action", "trained_parameters": list(MASK_TRAINED_PARAMETERS), "value_function": "not trained",
                   "seeds": list(PPO_SEEDS), "same_dataset_for_all_seeds": True, "log_std_after_bc": LOG_STD_INIT, "initial_action_std": float(np.exp(LOG_STD_INIT))},
        "bc": BC, "relabeling": {"rounds": 1, "scenarios": [RELABEL_SEED_START, RELABEL_SEED_START + RELABEL_EPISODES - 1], "max_states_per_episode": RELABEL_MAX_STEPS,
                                 "policy": "round-0 deterministic policy mean", "label": "A0 action at the visited state", "fine_tune": FINE_TUNE, "adaptive_repetition": False},
        "evaluation": {"held_out": [C.VALIDATION_A[0], C.VALIDATION_B[1]], "episodes": 200, "policy": "deterministic mean, no A0 fallback", "never_used_for": "BC or relabeling"},
        "gate": GATE_V2, "pass_rule": ">= 2/3 seeds pass AND no shared structural pathology (same pathology category in > 1% of episodes for >= 2 passing seeds)",
        "no_ppo": "no PPO learning call is ever made; PPO objects are constructed only to hold the policy architecture",
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()
