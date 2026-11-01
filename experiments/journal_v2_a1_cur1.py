"""Curriculum Stage 1 for the V2 learned attacker A1: reach and capture a STATIONARY asset in open space (C0, no active defender).

Identifiers: action `a1_cur1_action_v1` (continuous desired 3-D direction), observation `a1_v2_obs_v1` (37-D, unchanged), reward `a1_v2_reward_v2` (unchanged),
training protocol `a1_cur1_training_v1`. See docs/journal_v2_a1_curriculum/stage1/.

Everything here is configuration / wrapper only: `uav_defend` is untouched (byte-identical to the canonical A0 tag).

  * asset      : `v_s = 0` in the (otherwise canonical) C0 V2 config  -> the asset never moves after reset. (The canonical reset places it at the origin.)
  * defender   : established passive/parked semantics (`defender_standby_until_detection=False`, never acts, state kept in the simulator), parked at a GROUND-LEVEL
                 arena corner (-48, -48, 0): every attacker spawn is >= 10 m above it, straight corridors to the ground asset leave it behind, and the 25 m information
                 gate is open only in ~7.5% of episodes (stationary, non-threatening defender features); A0 never gets closer than 9 m (600 episodes).
  * clutter    : C0 (no static / dynamic obstacles; obstacle avoidance disabled by the canonical C0 config).
  * geometry   : canonical reset distribution (attacker on the 50 m perimeter, 10-30 m altitude) PLUS a pre-registered random initial heading (below).
  * action     : a in [-1,1]^3 -> u = a/|a| is the desired direction handed to the canonical constrained dynamics through the existing
                 `enemy_tactical_override` hook. |a| < 1e-6 or non-finite -> HOLD the current velocity direction (+x if the vehicle is at rest). This is
                 deliberately NOT the environment's own fallback (which would silently return the A0 direct-pursuit command).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.sanitize import build_policy_info  # noqa: F401  (kept for parity with the other A1 envs; defender never acts here)

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_r2 as R
from experiments.journal_v2_validation_core import build_config, count_overshoots, orbit_revolutions

CURRICULUM = "journal_v2_a1_curriculum"
STAGE = "stage1"
ACTION_VERSION = "a1_cur1_action_v1"
OBSERVATION_VERSION = A.A1_V2_OBSERVATION_VERSION      # "a1_v2_obs_v1"
REWARD_VERSION = R.REWARD_VERSION                        # "a1_v2_reward_v2"
TRAINING_PROTOCOL = "a1_cur1_training_v1"
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_curriculum" / STAGE

# ----------------------------------------------------------------------------- pre-registered constants (frozen before any training)
SCENARIO_STREAM_TAG = 0xC1A1          # training scenario stream (never used elsewhere)
HEADING_STREAM_TAG = 0xC1A2           # per-scenario initial-heading stream
MAX_HEADING_OFFSET_DEG = 90.0         # initial velocity direction is rotated away from the line of sight by an angle ~ U(0, 90 deg), uniform azimuth
TRAIN_SEED_LOW, TRAIN_SEED_HIGH = 1_000_000, 2**31 - 1
VALIDATION_A = (146_000, 146_099)
VALIDATION_B = (146_100, 146_199)
PARKED_DEFENDER = (-48.0, -48.0, 0.0)  # established parked semantics at a ground-level arena corner (the env clamps any defender into the 50 m volume, so "outside" is impossible)
HOLD_EPS = 1e-6
N_ENVS = A.A1_V2_NUM_ENVS
MAX_ROLLOUTS = 8
PPO_SEEDS = A.A1_V2_PPO_SEEDS        # 201, 202, 203
NONCLOSING_STEPS = 10                # >= 10 consecutive steps with closing rate < 1 m/s = "prolonged non-closing"
OVERSHOOT_PATHOLOGY = 2              # >= 2 approach->recede transitions = repeated fly-bys
ORBIT_PATHOLOGY_REV = 1.0            # >= 1 bearing revolution about the asset within 20 m

GATE = {  # mastery gate (frozen): a seed is mastered after TWO CONSECUTIVE rollout validations (alternating sets A/B) that each satisfy ALL of these
    "capture_min": 0.98, "collision_max": 0.0, "exhaustion_max": 0.01, "timeout_max": 0.01, "median_path_excess_max": 1.20,
    "gt2_max": 0.02, "gt5_max": 0.0, "pathology_rate_max": 0.02, "orbit_episodes_max": 0, "nonfinite_max": 0,
}
VALIDATION_ROLLOUT_SET = {True: "A", False: "B"}   # rollout k: odd -> A, even -> B


def validation_seeds(name: str) -> range:
    lo, hi = {"A": VALIDATION_A, "B": VALIDATION_B}[name]
    return range(lo, hi + 1)


def final_evaluation_seeds() -> list[int]:
    return list(validation_seeds("A")) + list(validation_seeds("B"))


def stage1_config():
    """Canonical C0 V2 config (passive-defender harness variant) with the asset frozen (`v_s = 0`). Nothing else differs from canonical."""
    config, _ = build_config("C0", "V2", "passive")
    return dataclasses.replace(config, v_s=0.0)


def config_differences_from_canonical() -> dict:
    canonical = A.build_episode_config("C0", "greedy")
    stage = stage1_config()
    out = {}
    for f in dataclasses.fields(canonical):
        a, b = getattr(canonical, f.name), getattr(stage, f.name)
        if a != b:
            out[f.name] = {"canonical": a, "stage1": b}
    return out


def config_sha256(config=None) -> str:
    config = stage1_config() if config is None else config
    return hashlib.sha256(json.dumps(dataclasses.asdict(config), sort_keys=True, default=str).encode()).hexdigest()


def scenario_stream(ppo_seed: int, worker_index: int, rollout_index: int = 0) -> np.random.Generator:
    """Training scenario RNG: a pure function of (ppo_seed, worker, rollout index) so an interrupted run replays exactly the scenarios an uninterrupted run sees."""
    return np.random.default_rng(np.random.SeedSequence([SCENARIO_STREAM_TAG, int(ppo_seed), int(worker_index), int(rollout_index)]))


def sample_scenario_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(TRAIN_SEED_LOW, TRAIN_SEED_HIGH))


def initial_heading_direction(scenario_seed: int, line_of_sight: np.ndarray) -> np.ndarray:
    """Unit initial velocity direction: the line of sight rotated by alpha ~ U(0, MAX_HEADING_OFFSET_DEG) about a uniformly random azimuth."""
    rng = np.random.default_rng(np.random.SeedSequence([HEADING_STREAM_TAG, int(scenario_seed)]))
    alpha = math.radians(MAX_HEADING_OFFSET_DEG) * float(rng.uniform())
    phi = 2.0 * math.pi * float(rng.uniform())
    los = np.asarray(line_of_sight, dtype=np.float64)
    los = los / np.linalg.norm(los)
    helper = np.array([0.0, 0.0, 1.0]) if abs(los[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(los, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(los, e1)
    d = math.cos(alpha) * los + math.sin(alpha) * (math.cos(phi) * e1 + math.sin(phi) * e2)
    return d / np.linalg.norm(d)


def desired_direction(action, velocity) -> tuple[np.ndarray, bool]:
    """(unit direction, held). a/|a| for a usable action; otherwise HOLD the current velocity direction (+x at rest). Never the A0 pursuit command."""
    raw = np.asarray(action, dtype=np.float64).reshape(-1)
    raw = np.clip(raw, -1.0, 1.0) if np.isfinite(raw).all() else raw
    norm = float(np.linalg.norm(raw)) if np.isfinite(raw).all() else float("nan")
    if np.isfinite(norm) and norm >= HOLD_EPS:
        return raw / norm, False
    v = np.asarray(velocity, dtype=np.float64)
    speed = float(np.linalg.norm(v))
    return (v / speed if speed > HOLD_EPS else np.array([1.0, 0.0, 0.0])), True


class Stage1Env(gym.Env):
    metadata: dict = {"render_modes": []}

    def __init__(self, ppo_seed: int, worker_index: int = 0, rollout_index: int = 0, scripted_a0: bool = False, record_trajectory: bool = False,
                 gamma: float = R.GAMMA):
        super().__init__()
        self.ppo_seed, self.worker_index = int(ppo_seed), int(worker_index)
        self._scripted_a0, self.record_trajectory, self.gamma = bool(scripted_a0), bool(record_trajectory), float(gamma)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(A.A1_V2_OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._config = stage1_config()
        self._stream = scenario_stream(ppo_seed, worker_index, rollout_index)
        self._env: SoldierEnv | None = None
        self._pending = np.zeros(3, dtype=np.float32)
        self.scenario_seed: int | None = None

    def reseed(self, rollout_index: int) -> int:
        """Replace the scenario stream by the one of `rollout_index` (called by the runner at every rollout boundary)."""
        self._stream = scenario_stream(self.ppo_seed, self.worker_index, rollout_index)
        return int(rollout_index)

    @property
    def soldier_env(self) -> SoldierEnv | None:
        return self._env

    def _override(self) -> np.ndarray:
        direction, held = desired_direction(self._pending, self._env._enemy_vel)
        self._held_steps += int(held)
        self._invalid_actions += int(not np.isfinite(np.asarray(self._pending, dtype=np.float64)).all())
        return direction

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.scenario_seed = int(options["scenario"]) if options and "scenario" in options else sample_scenario_seed(self._stream)
        self._env = SoldierEnv(self._config)
        self._env.enemy_tactical_override = None if self._scripted_a0 else self._override
        self._env.reset(seed=self.scenario_seed)
        env = self._env
        env._defender_pos = np.asarray(PARKED_DEFENDER, dtype=np.float32)
        env._defender_vel = np.zeros(3, dtype=np.float32)
        los = np.asarray(env._soldier_pos, dtype=np.float64) - np.asarray(env._enemy_pos, dtype=np.float64)
        direction = initial_heading_direction(self.scenario_seed, los)
        velocity = float(self._config.v_e) * direction
        velocity[2] = np.clip(velocity[2], -self._config.enemy_max_descent_rate, self._config.enemy_max_climb_rate)  # same invariant the canonical reset keeps
        env._enemy_vel = velocity.astype(np.float32)
        self._pending = np.zeros(3, dtype=np.float32)
        self._held_steps = self._invalid_actions = 0
        self._detection_range = float(self._config.enemy_defender_detection_range)
        self._max_altitude, self._dt = float(self._config.max_altitude), float(self._config.dt)
        self._start_pos = np.asarray(env._enemy_pos, dtype=np.float64).copy()
        self._asset0 = np.asarray(env._soldier_pos, dtype=np.float64).copy()
        self._prev_pos = self._start_pos.copy()
        self._range = self._range_now()
        self._initial_range = self._range
        self._initial_alt = float(self._start_pos[2])
        self._initial_offset_deg = float(np.degrees(np.arccos(np.clip(float(np.dot(direction, los / np.linalg.norm(los))), -1.0, 1.0))))
        self._path_prev = float(env._enemy_endurance_traveled)
        self._budget = float(env._enemy_endurance_budget)
        self._steps = 0
        self._sum = dict(terminal=0.0, progress=0.0, path=0.0, step=0.0)
        self._discounted = 0.0
        self._ranges, self._alts, self._speeds, self._att, self._ast = [], [], [], [], []
        self._asset_moved = 0.0
        self._min_def_sep = float("inf")
        self._traj = [self._start_pos.round(2).tolist()] if self.record_trajectory else []
        return self._observation(), self._info({}, None)

    def _observation(self) -> np.ndarray:
        return A.build_a1_v2_observation(self._env, self._detection_range).astype(np.float32)

    def _range_now(self) -> float:
        return float(np.linalg.norm(np.asarray(self._env._soldier_pos, dtype=np.float64) - np.asarray(self._env._enemy_pos, dtype=np.float64)))

    def step(self, action):
        self._pending = np.asarray(action, dtype=np.float32).copy()
        _o, _r, terminated, truncated, info = self._env.step(np.zeros(3, dtype=np.float32))   # the parked defender never acts
        self._steps += 1
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        self._speeds.append(float(np.linalg.norm(pos - self._prev_pos)) / self._dt)
        self._alts.append(float(pos[2]))
        self._att.append(pos)
        self._ast.append(np.asarray(info["soldier_pos"], dtype=np.float64))
        self._asset_moved = max(self._asset_moved, float(np.linalg.norm(np.asarray(info["soldier_pos"], dtype=np.float64) - self._asset0)))
        self._min_def_sep = min(self._min_def_sep, float(info["defender_enemy_dist"]))
        range_next = self._range_now()
        self._ranges.append(range_next)
        if self.record_trajectory:
            self._traj.append(pos.round(2).tolist())
        self._prev_pos = pos
        path_now = float(self._env._enemy_endurance_traveled)
        outcome = info["outcome"]
        reward = R.step_reward(range_prev_m=self._range, range_next_m=range_next, initial_range_m=self._initial_range,
                               path_step_m=path_now - self._path_prev, path_budget_m=self._budget, outcome=outcome, gamma=self.gamma)
        self._range, self._path_prev = range_next, path_now
        for key in self._sum:
            self._sum[key] += getattr(reward, key)
        self._discounted += (self.gamma ** (self._steps - 1)) * reward.total
        episode = self._episode_summary(outcome, path_now, range_next) if (terminated or truncated) else None
        return self._observation(), float(reward.total), bool(terminated), bool(truncated), self._info(info, reward, episode)

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        chord = float(np.linalg.norm(self._prev_pos - self._start_pos))
        ranges = np.array([self._initial_range] + self._ranges)
        rate = np.diff(ranges) / self._dt
        run = longest = 0
        for r in rate:
            run = run + 1 if r > -1.0 else 0   # closing slower than 1 m/s
            longest = max(longest, run)
        orbit_max, _ = orbit_revolutions(np.array(self._att), np.array(self._ast))
        overshoots = int(count_overshoots(ranges, self._dt))
        finite = bool(np.isfinite(np.array(self._att)).all() and np.isfinite(ranges).all() and np.isfinite(path_now) and np.isfinite(self._discounted))
        pathology = bool(orbit_max >= ORBIT_PATHOLOGY_REV or overshoots >= OVERSHOOT_PATHOLOGY or longest >= NONCLOSING_STEPS)
        out = dict(
            outcome=outcome, scenario_seed=self.scenario_seed, steps=self._steps, duration_s=self._steps * self._dt,
            return_undiscounted=sum(self._sum.values()), return_discounted=self._discounted, components=dict(self._sum),
            path_length_m=path_now, chord_m=chord, path_excess=(path_now / chord) if chord > 1e-9 else float("nan"),
            initial_range_m=self._initial_range, terminal_distance_m=range_next, progress_net=1.0 - range_next / self._initial_range,
            endurance_exhausted=outcome == "enemy_endurance_exhausted", orbit_revolutions_max=float(orbit_max), overshoots=overshoots,
            nonclosing_run=int(longest), pathology=pathology, finite=finite, held_steps=int(self._held_steps), invalid_actions=int(self._invalid_actions),
            asset_displacement_max_m=float(self._asset_moved), min_defender_separation_m=float(self._min_def_sep),
            initial_altitude_m=self._initial_alt, initial_heading_offset_deg=self._initial_offset_deg,
            mean_altitude_m=float(np.mean(self._alts)), mean_speed_mps=float(np.mean(self._speeds)),
        )
        if self.record_trajectory:
            out["trajectory"] = self._traj
        return out

    def _info(self, info: dict, reward, episode: dict | None = None) -> dict:
        info = dict(info)
        info["a1_reward_version"], info["a1_action_version"], info["a1_observation_version"] = REWARD_VERSION, ACTION_VERSION, OBSERVATION_VERSION
        if reward is not None:
            info["a1_reward_components"] = {"terminal": reward.terminal, "progress": reward.progress, "path": reward.path, "step": reward.step}
        if episode is not None:
            info["a1_episode"] = episode
        return info

    def close(self):
        self._env = None


def make_env(ppo_seed: int, worker_index: int, rollout_index: int = 0) -> Stage1Env:
    return Stage1Env(ppo_seed, worker_index, rollout_index)


# ----------------------------------------------------------------------------- evaluation + mastery gate
def run_episode(env: Stage1Env, scenario_seed: int, act) -> dict:
    """One deterministic episode; `act(obs) -> action` (None for scripted A0, which ignores actions)."""
    obs, _ = env.reset(options={"scenario": int(scenario_seed)})
    done, info = False, {}
    while not done:
        action = np.zeros(3, dtype=np.float32) if act is None else np.asarray(act(obs), dtype=np.float32)
        obs, _r, te, tr, info = env.step(action)
        done = te or tr
    return info["a1_episode"]


def evaluate(act, seeds, *, scripted_a0: bool = False, record_trajectory: bool = False) -> list[dict]:
    env = Stage1Env(0, 0, 0, scripted_a0=scripted_a0, record_trajectory=record_trajectory)
    return [run_episode(env, s, None if scripted_a0 else act) for s in seeds]


def model_actor(model):
    def act(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return act


def metrics(episodes: list[dict]) -> dict:
    n = len(episodes)
    pe = np.array([e["path_excess"] for e in episodes], dtype=np.float64)
    frac = lambda mask: float(np.mean(mask))  # noqa: E731
    outcome = np.array([e["outcome"] for e in episodes])
    collisions = np.isin(outcome, ["enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"])
    out = {
        "episodes": n, "capture": frac(outcome == "soldier_caught"), "exhaustion": frac(outcome == "enemy_endurance_exhausted"), "timeout": frac(outcome == "timeout"),
        "collision": frac(collisions), "intercepted": frac(np.isin(outcome, ["intercepted", "unsafe_intercept"])),
        "median_path_excess": float(np.nanmedian(pe)), "mean_path_excess": float(np.nanmean(pe)), "max_path_excess": float(np.nanmax(pe)),
        "gt1.3": frac(pe > 1.3), "gt2": frac(pe > 2.0), "gt5": frac(pe > 5.0), "gt5_count": int((pe > 5.0).sum()),
        "orbit_episodes": int(sum(e["orbit_revolutions_max"] >= ORBIT_PATHOLOGY_REV for e in episodes)),
        "flyby_episodes": int(sum(e["overshoots"] >= 1 for e in episodes)), "repeated_flyby_episodes": int(sum(e["overshoots"] >= OVERSHOOT_PATHOLOGY for e in episodes)),
        "nonclosing_episodes": int(sum(e["nonclosing_run"] >= NONCLOSING_STEPS for e in episodes)), "pathology_rate": frac([e["pathology"] for e in episodes]),
        "nonfinite": int(sum(not e["finite"] or e["invalid_actions"] > 0 for e in episodes)), "held_action_steps": int(sum(e["held_steps"] for e in episodes)),
        "mean_duration_s": float(np.mean([e["duration_s"] for e in episodes])), "mean_path_length_m": float(np.mean([e["path_length_m"] for e in episodes])),
        "mean_terminal_distance_m": float(np.mean([e["terminal_distance_m"] for e in episodes])), "mean_return": float(np.mean([e["return_undiscounted"] for e in episodes])),
        "mean_return_discounted": float(np.mean([e["return_discounted"] for e in episodes])),
        "asset_displacement_max_m": float(max(e["asset_displacement_max_m"] for e in episodes)),
        "min_defender_separation_m": float(min(e["min_defender_separation_m"] for e in episodes)),
    }
    for key in ("terminal", "progress", "path", "step"):
        out[f"reward_{key}"] = float(np.mean([e["components"][key] for e in episodes]))
    return out


def gate_status(m: dict) -> dict:
    checks = {
        "capture": m["capture"] >= GATE["capture_min"], "collision": m["collision"] <= GATE["collision_max"], "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"],
        "timeout": m["timeout"] <= GATE["timeout_max"], "median_path_excess": m["median_path_excess"] <= GATE["median_path_excess_max"],
        "gt2": m["gt2"] <= GATE["gt2_max"], "gt5": m["gt5"] <= GATE["gt5_max"], "pathology": m["pathology_rate"] <= GATE["pathology_rate_max"],
        "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"], "numerical_stability": m["nonfinite"] <= GATE["nonfinite_max"],
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def protocol_description() -> dict:
    return {"training_protocol": TRAINING_PROTOCOL, "reward_version": REWARD_VERSION, "action_version": ACTION_VERSION, "observation_version": OBSERVATION_VERSION,
            "curriculum": CURRICULUM, "stage": STAGE}


def protocol_specification() -> dict:
    """The complete pre-registered Stage-1 protocol (hashed in the task state; any change after training starts invalidates the run)."""
    return {
        "hypothesis": "PPO can learn reliable asset capture under canonical attacker dynamics when the asset is stationary and defender/clutter complexity is removed.",
        "identifiers": protocol_description(), "env_config_sha256": config_sha256(), "env_config_differences_from_canonical_C0_V2": config_differences_from_canonical(),
        "asset": "v_s = 0 (stationary after reset); canonical reset places it at the origin on the ground; relative geometry randomized through the attacker spawn + heading",
        "defender": {"semantics": "established passive/parked: standby disabled, never acts, state retained", "position": list(PARKED_DEFENDER),
                     "note": "the environment clamps any defender into the 50 m volume; ground corner chosen to minimize interference (A0 min separation 9.0 m over 600 episodes, "
                             "defender visible within 25 m in ~7.5% of episodes, stationary)"},
        "clutter": "C0",
        "training_geometry": {"attacker_spawn": "canonical: uniform edge of the 50 m square perimeter, uniform position along the edge, altitude U(10, 30) m",
                              "asset_to_attacker_distance_m": "~50-75 (3-D, canonical)", "initial_heading": f"line of sight rotated by alpha ~ U(0, {MAX_HEADING_OFFSET_DEG}) deg, uniform azimuth; vertical rate clipped as in the canonical reset",
                              "initial_speed": "v_e = 12 m/s (canonical)", "scenario_stream": f"SeedSequence([{hex(SCENARIO_STREAM_TAG)}, ppo_seed, worker, rollout]); seeds in [{TRAIN_SEED_LOW}, {TRAIN_SEED_HIGH})",
                              "heading_stream": f"SeedSequence([{hex(HEADING_STREAM_TAG)}, scenario_seed])"},
        "action": {"space": "Box(-1, 1, (3,))", "mapping": "u = a/|a|; |a| < 1e-6 or non-finite -> hold the current velocity direction (+x at rest); never A0 guidance",
                   "speed_accel_turn_climb": "canonical constrained dynamics"},
        "observation": {"version": OBSERVATION_VERSION, "dim": A.A1_V2_OBSERVATION_DIM}, "reward": {"version": REWARD_VERSION, "special_stage_reward": False},
        "ppo": {"policy": A.A1_V2_POLICY, "hyperparameters": A.A1_V2_PPO_HYPERPARAMS, "n_envs": N_ENVS, "vec_env": "SubprocVecEnv", "seeds": list(PPO_SEEDS), "tuning": "none"},
        "training": {"steps_per_rollout": N_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"], "max_rollouts": MAX_ROLLOUTS, "max_timesteps": MAX_ROLLOUTS * N_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"],
                     "per_rollout_rekey_and_reset": True},
        "validation": {"set_A": list(VALIDATION_A), "set_B": list(VALIDATION_B), "rollout_odd": "A", "rollout_even": "B", "policy": "deterministic (mean action)",
                       "reserved_ranges_checked": "no overlap with 130000-130099, 140000-140099, 145000-145099, 150000-150039, 180000-180999, 190000-190999, 95000-95999"},
        "mastery_gate": GATE,
        "pathology_definition": {"orbit_revolutions_ge": ORBIT_PATHOLOGY_REV, "repeated_flybys_ge": OVERSHOOT_PATHOLOGY, "nonclosing_steps_ge": NONCLOSING_STEPS,
                                 "non_closing_step": "closing rate < 1 m/s", "gate": "pathology rate <= 2% and zero orbit episodes"},
        "early_stopping": "stop a seed at the second consecutive passing validation; FAIL if not mastered by rollout 8",
        "stage_pass": ">= 2 of 3 seeds mastered, every mastered seed passes the final 200-episode held-out gates, no common structural pathology",
        "final_evaluation": {"episodes_per_seed": 200, "seeds": [VALIDATION_A[0], VALIDATION_B[1]], "a0_reference": "scripted canonical A0 on the identical scenarios"},
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()
