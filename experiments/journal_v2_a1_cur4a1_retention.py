"""Stage3-v2 behaviour-RETENTION (rehearsal) dataset for curriculum Stage 4A-1: driven by the SEED'S OWN accepted Stage3-v2 parent policy (not A0) on reserved Stage3-v2 (hold-until-detection
Greedy, stationary asset) scenarios. At every decision we record the observation and the PARENT policy's deterministic mean action (the retention TARGET; never a future-outcome label). States are
classified post-hoc (after the episode outcome is known, for REPORTING only -- never fed into the label) into `not_visible` / `newly_visible` / `active_evasion` / `post_pass_pursuit` using the
episode's own `detection_step` and `closest_step` (the step of the minimum attacker-defender distance)."""

from __future__ import annotations

import hashlib

import numpy as np

from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur4a1 as Z

STATE_CLASSES = ("not_visible", "newly_visible", "active_evasion", "post_pass_pursuit")


def classify(step: int, detection_step: int | None, closest_step: int) -> str:
    if detection_step is None or step < detection_step:
        return "not_visible"
    if step == detection_step:
        return "newly_visible"
    return "active_evasion" if step <= closest_step else "post_pass_pursuit"


def collect_retention_dataset(model, target_states: int = Z.RETENTION_TARGET_PER_SEED, seeds=None) -> dict:
    """Runs the PARENT model's own deterministic policy (not A0) on `seeds` (default: `RETENTION_SOURCE`, in order) and stops as soon as `target_states` decisions have been collected."""
    import torch

    seeds = list(seeds if seeds is not None else Z.seeds_of(Z.RETENTION_SOURCE))
    env = X.Stage3V2Env(0, 0, 0)
    obs_rows, act_rows, cls_rows, ep_seed, step_idx = [], [], [], [], []
    episodes_used = 0
    for s in seeds:
        if len(obs_rows) >= target_states:
            break
        obs, _ = env.reset(options={"scenario": int(s)})
        episode_obs, episode_act, done, info = [], [], False, {}
        while not done:
            with torch.no_grad():
                action = BC.mean_action(model, torch.as_tensor(obs[None, :])).numpy()[0]
            episode_obs.append(obs.copy())
            episode_act.append(action.copy())
            obs, _r, te, tr, info = env.step(action)
            done = te or tr
        ep = info["a1_episode"]
        detect, closest = ep["detection_step"], ep["closest_step"]
        for t, (o, a) in enumerate(zip(episode_obs, episode_act)):
            obs_rows.append(o)
            act_rows.append(a)
            cls_rows.append(classify(t, detect, closest))
            ep_seed.append(int(s))
            step_idx.append(t)
        episodes_used += 1
    obs_arr, act_arr = np.asarray(obs_rows[:target_states], dtype=np.float32), np.asarray(act_rows[:target_states], dtype=np.float32)
    cls_arr, ep_arr, step_arr = np.asarray(cls_rows[:target_states]), np.asarray(ep_seed[:target_states], dtype=np.int64), np.asarray(step_idx[:target_states], dtype=np.int16)
    return {"obs": obs_arr, "act": act_arr, "class": cls_arr, "ep_seed": ep_arr, "step": step_arr, "episodes_used": episodes_used, "scenario_range": [seeds[0], seeds[episodes_used - 1] if episodes_used else seeds[0]]}


def retention_coverage(ds: dict) -> dict:
    n = len(ds["obs"])
    return {"states": n, "episodes_used": ds["episodes_used"], "scenario_range": ds["scenario_range"],
            "class_fractions": {c: float(np.mean(ds["class"] == c)) for c in STATE_CLASSES}, "class_counts": {c: int(np.sum(ds["class"] == c)) for c in STATE_CLASSES}}


def dataset_sha256(ds: dict) -> str:
    return hashlib.sha256(ds["obs"].tobytes() + ds["act"].tobytes()).hexdigest()


def split(ds: dict, fraction: float = 0.8) -> dict:
    """Episode-level split, reusing the Stage-1-v2 permutation mechanism keyed by the retention source range (no episode leakage)."""
    from experiments import journal_v2_a1_cur1v2 as V1V2

    lo, hi = Z.RETENTION_SOURCE
    flags = V1V2.split_table(lo, hi - lo + 1, fraction, extra=2)
    is_train = flags[ds["ep_seed"] - lo]
    return {"train": (ds["obs"][is_train], ds["act"][is_train]), "val": (ds["obs"][~is_train], ds["act"][~is_train])}
