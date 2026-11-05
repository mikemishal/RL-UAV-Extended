"""Combined goal-imitation + Stage3-v2 behaviour-retention supervised fine-tuning for curriculum Stage 4A-1. NO PPO `learn()` anywhere in this module.

* starts from the seed's accepted Stage3-v2 checkpoint (actor + critic + log_std copied exactly), then forces `log_std := -2.0` and freezes it (only `mlp_extractor.policy_net` and `action_net`
  receive gradients -- the critic and `log_std` are never updated here);
* total loss per minibatch = L_goal(mean, a_A0) + 0.25 * L_retain(mean, a_parent), each of the form MSE + (1 - cos); the retention minibatch is drawn (with replacement across epochs if smaller than
  the goal set) from the seed-specific Stage3-v2 rehearsal dataset;
* every epoch is committed atomically (checkpoint then hashed manifest LAST); shuffling is a pure function of (seed, epoch), so an interrupted run resumes bit-exactly.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur4a1 as Z
from experiments.journal_v2_a1_eg_state import CellLock

T = BC  # re-exported low-level primitives (actor_state, tensor_sha256, epoch_names, verify_epoch, commit_epoch, recover, mean_action, set_log_std, save_model, load_model, model_actor)
RunConflictError = RuntimeError


def initialize_model(seed: int, parent: dict | None = None):
    """Fresh PPO architecture, actor+critic+log_std OVERWRITTEN from the Stage3-v2 parent, then log_std forced to -2.0 (frozen; never trained here)."""
    from stable_baselines3 import PPO

    parent = parent or Z.parent_checkpoint(seed)
    model = BC.build_model(seed)
    src = PPO.load(parent["path"].removesuffix(".zip"), device="cpu")
    model.policy.load_state_dict(src.policy.state_dict())
    if BC.tensor_sha256(BC.actor_state(model)) != BC.tensor_sha256(BC.actor_state(src)) or BC.value_state_sha256(model) != BC.value_state_sha256(src):
        raise RunConflictError("actor / critic were not copied exactly from the Stage3-v2 parent")
    BC.set_log_std(model, -2.0)
    record = {"seed": int(seed), "parent": {k: parent[k] for k in ("attacker_id", "path", "model_sha256", "rollout", "stage3_v2_training_status", "parent_status", "stage3_v2_commit")},
              "parent_status": Z.PARENT_STATUS, "copied": ["actor (policy_net, action_net)", "critic (value_net)"], "log_std_forced_to": -2.0, "log_std": [float(x) for x in model.policy.log_std.detach().numpy()],
              "trained_parameters": ["mlp_extractor.policy_net", "action_net"], "frozen": ["value_net", "log_std", "features_extractor"]}
    return model, record


def combined_metrics(model, goal_val: tuple, retain_val: tuple) -> dict:
    goal = BC.bc_metrics(model, *goal_val)
    retain = BC.bc_metrics(model, *retain_val)
    return {"goal": goal, "retain": retain, "loss": goal["loss"] + Z.RETAIN_LOSS_WEIGHT * retain["loss"]}


def epoch_names(e: int) -> dict:
    return BC.epoch_names(e)


def verify_epoch(work_dir: Path, e: int, *, seed: int, tag: str) -> dict | None:
    return BC.verify_epoch(work_dir, e, seed=seed, tag=tag)


def recover(work_dir: Path, epochs: int, *, seed: int, tag: str) -> int:
    return BC.recover(work_dir, epochs, seed=seed, tag=tag)


def commit_epoch(work_dir: Path, e: int, model, optimizer, history: list, *, seed: int, tag: str, hyper: dict, fault_hook=None) -> dict:
    return BC.commit_epoch(work_dir, e, model, optimizer, history, seed=seed, tag=tag, hyper=hyper, fault_hook=fault_hook)


def train_combined(model, goal_train: tuple, retain_train: tuple, goal_val: tuple, retain_val: tuple, *, work_dir: Path, seed: int, tag: str, hyper: dict, fault_hook=None, stop_after: int | None = None) -> dict:
    """Actor-only fine-tuning for `hyper['epochs']` epochs with per-epoch atomic checkpoints, resuming from the latest verified epoch. Returns the history and the SELECTED epoch (lowest combined
    validation loss); the model holds the selected weights afterwards."""
    import torch

    lock = CellLock(Path(work_dir) / "train")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    if not lock.acquire():
        raise RunConflictError(f"another process is training {work_dir}")
    try:
        epochs, lr, batch = int(hyper["epochs"]), float(hyper["lr"]), int(hyper["batch_size"])
        g_obs, g_act = torch.as_tensor(goal_train[0]), torch.as_tensor(goal_train[1])
        r_obs, r_act = torch.as_tensor(retain_train[0]), torch.as_tensor(retain_train[1])
        optimizer = torch.optim.Adam(BC.actor_parameters(model), lr=lr)
        done = recover(work_dir, epochs, seed=seed, tag=tag)
        history: list = []
        if done:
            state = torch.load(Path(work_dir) / epoch_names(done)["ckpt"], weights_only=False)
            BC.load_actor_state(model, state["actor"])
            optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
        n, m = g_obs.shape[0], r_obs.shape[0]
        for e in range(done + 1, epochs + 1):
            g = torch.Generator().manual_seed(1_000_007 * int(seed) + e)
            g_perm = torch.randperm(n, generator=g)
            r_gen = torch.Generator().manual_seed(2_000_003 * int(seed) + e)
            r_perm = torch.randperm(m, generator=r_gen)
            model.policy.set_training_mode(True)
            total, count = 0.0, 0
            for i in range(0, n, batch):
                g_idx = g_perm[i:i + batch]
                r_idx = r_perm[torch.arange(i, i + len(g_idx)) % m]
                g_mean = BC.mean_action(model, g_obs[g_idx])
                r_mean = BC.mean_action(model, r_obs[r_idx])
                g_tgt, r_tgt = g_act[g_idx], r_act[r_idx]
                loss_goal = ((g_mean - g_tgt) ** 2).mean() + (1.0 - torch.nn.functional.cosine_similarity(g_mean, g_tgt, dim=1, eps=1e-8).mean())
                loss_retain = ((r_mean - r_tgt) ** 2).mean() + (1.0 - torch.nn.functional.cosine_similarity(r_mean, r_tgt, dim=1, eps=1e-8).mean())
                loss = loss_goal + Z.RETAIN_LOSS_WEIGHT * loss_retain
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(g_idx)
                count += len(g_idx)
            model.policy.set_training_mode(False)
            history.append({"epoch": e, "train_loss": total / count, "val": combined_metrics(model, goal_val, retain_val)})
            commit_epoch(work_dir, e, model, optimizer, history, seed=seed, tag=tag, hyper=hyper, fault_hook=fault_hook)
            if stop_after is not None and e >= stop_after:
                return {"history": history, "completed_epochs": e, "selected_epoch": None}
        best = min(history, key=lambda r: (r["val"]["loss"], r["epoch"]))["epoch"]
        state = torch.load(Path(work_dir) / epoch_names(best)["ckpt"], weights_only=False)
        BC.load_actor_state(model, state["actor"])
        return {"history": history, "completed_epochs": epochs, "selected_epoch": best}
    finally:
        lock.release()
