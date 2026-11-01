"""Behavior cloning of the A0 expert into the exact PPO policy architecture (SB3 MlpPolicy, [64,64] tanh, 37-D -> 3-D Gaussian). NO PPO learn().

* trained parameters: `policy.mlp_extractor.policy_net` and `policy.action_net` (the policy mean); the value network, `log_std` and the features extractor are NOT trained;
* loss = MSE(mean, a_A0) + (1 - cos(mean, a_A0)); reported: MSE, mean / median / 5th-percentile cosine (primary criterion: cosine -> 1);
* after BC (and after the single fine-tuning round) `log_std := -2.0` (initial action std e^-2 = 0.135) for later PPO curriculum learning;
* every epoch is committed atomically (checkpoint file then hashed manifest LAST); shuffling is a pure function of (seed, epoch), so an interrupted run resumes bit-exactly.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V
from experiments import journal_v2_a1_eg_state as P


class SimulatedPowerLoss(BaseException):
    """Raised only by fault-injection hooks in tests."""


def build_model(seed: int):
    """A fresh PPO object (constructed ONLY to hold the policy architecture / spaces; `learn()` is never called)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([functools.partial(V.make_env, seed, 0, 0)])
    return PPO(policy=A.A1_V2_POLICY, env=env, seed=seed, verbose=0, device="cpu", **A.A1_V2_PPO_HYPERPARAMS)


def actor_modules(model) -> list:
    return [model.policy.mlp_extractor.policy_net, model.policy.action_net]


def actor_parameters(model) -> list:
    return [p for m in actor_modules(model) for p in m.parameters()]


def actor_state(model) -> dict:
    return {f"{i}.{k}": v.detach().clone() for i, m in enumerate(actor_modules(model)) for k, v in m.state_dict().items()}


def load_actor_state(model, state: dict) -> None:
    for i, m in enumerate(actor_modules(model)):
        m.load_state_dict({k.split(".", 1)[1]: v for k, v in state.items() if k.startswith(f"{i}.")})


def tensor_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model) -> str:
    """Hash of every policy tensor (actor, critic, log_std)."""
    return tensor_sha256({k: v for k, v in model.policy.state_dict().items()})


def value_state_sha256(model) -> str:
    return tensor_sha256({f"v.{k}": v for k, v in model.policy.mlp_extractor.value_net.state_dict().items()} | {f"vo.{k}": v for k, v in model.policy.value_net.state_dict().items()})


def mean_action(model, obs_tensor):
    return model.policy.get_distribution(obs_tensor).distribution.mean


def bc_metrics(model, obs: np.ndarray, act: np.ndarray, batch: int = 8192) -> dict:
    import torch

    cos_all, sq, n = [], 0.0, 0
    with torch.no_grad():
        for i in range(0, len(obs), batch):
            o = torch.as_tensor(obs[i:i + batch])
            t = torch.as_tensor(act[i:i + batch])
            m = mean_action(model, o)
            cos_all.append(torch.nn.functional.cosine_similarity(m, t, dim=1, eps=1e-8))
            sq += float(((m - t) ** 2).sum())
            n += t.numel()
    cos = torch.cat(cos_all).numpy().astype(np.float64)
    mse = sq / n
    return {"samples": int(len(obs)), "mse": mse, "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)), "cos_p5": float(np.percentile(cos, 5)), "cos_min": float(cos.min()),
            "angle_deg_median": float(np.degrees(np.arccos(np.clip(np.median(cos), -1, 1)))), "loss": mse + (1.0 - float(cos.mean()))}


def epoch_names(e: int) -> dict:
    return {"ckpt": f"epoch_{e:03d}.pt", "manifest": f"epoch_{e:03d}.manifest.json"}


def verify_epoch(work_dir: Path, e: int, *, seed: int, tag: str) -> dict | None:
    names = epoch_names(e)
    try:
        manifest = json.loads((Path(work_dir) / names["manifest"]).read_text())
        if manifest.get("epoch") != e or manifest.get("seed") != seed or manifest.get("tag") != tag:
            return None
        if P.sha256_file(Path(work_dir) / names["ckpt"]) != manifest["ckpt_sha256"]:
            return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def commit_epoch(work_dir: Path, e: int, model, optimizer, history: list, *, seed: int, tag: str, hyper: dict, fault_hook=None) -> dict:
    import torch

    names = epoch_names(e)
    buffer = io.BytesIO()
    torch.save({"actor": actor_state(model), "optimizer": optimizer.state_dict(), "history": history}, buffer)
    payload = buffer.getvalue()
    P.atomic_write_bytes(Path(work_dir) / names["ckpt"], payload)
    if fault_hook is not None:
        fault_hook("before_manifest", e)
    manifest = {"epoch": e, "seed": seed, "tag": tag, "ckpt_sha256": hashlib.sha256(payload).hexdigest(), "actor_sha256": tensor_sha256(actor_state(model)),
                "val": history[-1]["val"], "train_loss": history[-1]["train_loss"], "hyper": hyper, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    P.atomic_write_json(Path(work_dir) / names["manifest"], manifest)
    if fault_hook is not None:
        fault_hook("after_manifest", e)
    return manifest


def recover(work_dir: Path, epochs: int, *, seed: int, tag: str) -> int:
    """Highest verified epoch (0 = none); temp files and every unverified/higher epoch artifact are removed."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for stale in work_dir.glob(".*.tmp*"):
        stale.unlink(missing_ok=True)
    best = 0
    for e in range(epochs, 0, -1):
        if verify_epoch(work_dir, e, seed=seed, tag=tag) is not None:
            best = e
            break
    for path in work_dir.glob("epoch_*"):
        number = int(path.name.split("_")[1].split(".")[0])
        if number > best or (number <= best and verify_epoch(work_dir, number, seed=seed, tag=tag) is None):
            path.unlink(missing_ok=True)
    return best


def train_bc(model, train: tuple, val: tuple, *, work_dir: Path, seed: int, tag: str, hyper: dict, fault_hook=None, stop_after: int | None = None) -> dict:
    """Train the actor for `hyper['epochs']` epochs with per-epoch atomic checkpoints, resuming from the latest verified epoch. Returns the history and the selected epoch
    (lowest validation loss); the model holds the SELECTED weights afterwards."""
    import torch

    lock = P.CellLock(Path(work_dir) / "train")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    if not lock.acquire():
        raise RuntimeError(f"another process is training {work_dir}")
    try:
        epochs, lr, batch = int(hyper["epochs"]), float(hyper["lr"]), int(hyper["batch_size"])
        obs_t, act_t = torch.as_tensor(train[0]), torch.as_tensor(train[1])
        optimizer = torch.optim.Adam(actor_parameters(model), lr=lr)
        done = recover(work_dir, epochs, seed=seed, tag=tag)
        history: list = []
        if done:
            state = torch.load(Path(work_dir) / epoch_names(done)["ckpt"], weights_only=False)
            load_actor_state(model, state["actor"])
            optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
        n = obs_t.shape[0]
        for e in range(done + 1, epochs + 1):
            g = torch.Generator().manual_seed(1_000_003 * int(seed) + e)
            perm = torch.randperm(n, generator=g)
            model.policy.set_training_mode(True)
            total, count = 0.0, 0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                m = mean_action(model, obs_t[idx])
                t = act_t[idx]
                loss = ((m - t) ** 2).mean() + (1.0 - torch.nn.functional.cosine_similarity(m, t, dim=1, eps=1e-8).mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss) * len(idx)
                count += len(idx)
            model.policy.set_training_mode(False)
            history.append({"epoch": e, "train_loss": total / count, "val": bc_metrics(model, *val)})
            commit_epoch(work_dir, e, model, optimizer, history, seed=seed, tag=tag, hyper=hyper, fault_hook=fault_hook)
            if stop_after is not None and e >= stop_after:
                return {"history": history, "completed_epochs": e, "selected_epoch": None}
        best = min(history, key=lambda r: (r["val"]["loss"], r["epoch"]))["epoch"]
        state = torch.load(Path(work_dir) / epoch_names(best)["ckpt"], weights_only=False)
        load_actor_state(model, state["actor"])
        return {"history": history, "completed_epochs": epochs, "selected_epoch": best}
    finally:
        lock.release()


def set_log_std(model, value: float = V.LOG_STD_INIT) -> None:
    model.policy.log_std.data.fill_(float(value))


def save_model(model, path: Path) -> str:
    buffer = io.BytesIO()
    model.save(buffer)
    payload = buffer.getvalue()
    P.atomic_write_bytes(Path(path), payload)
    return hashlib.sha256(payload).hexdigest()


def load_model(path: Path):
    from stable_baselines3 import PPO

    return PPO.load(str(path).removesuffix(".zip"), device="cpu")


def model_actor(model):
    def act(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return act


# ----------------------------------------------------------------------------- fixed one-round expert relabeling
def collect_visited_states(model, seeds, max_states: int = V.RELABEL_MAX_STEPS) -> dict:
    """Run the BC policy deterministically (its own actions drive the vehicle; no A0 intervention); at every visited state record the masked observation and the A0 label."""
    env = V.Stage1V2Env(0, 0, 0)
    obs_rows, act_rows, ep_seed = [], [], []
    outcomes: dict = {}
    steps_per_episode = []
    for s in seeds:
        obs, _ = env.reset(options={"scenario": int(s)})
        done, t, info = False, 0, {}
        while not done and t < max_states:
            obs_rows.append(obs.astype(np.float32))
            act_rows.append(env.expert_action())
            ep_seed.append(int(s))
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, te, tr, info = env.step(action)
            done, t = te or tr, t + 1
        steps_per_episode.append(t)
        key = info.get("outcome", "ongoing") if (done or t >= max_states) else "ongoing"
        outcomes["capped" if (not done and t >= max_states) else key] = outcomes.get("capped" if (not done and t >= max_states) else key, 0) + 1
    return {"obs": np.asarray(obs_rows, dtype=np.float32), "act": np.asarray(act_rows, dtype=np.float32), "ep_seed": np.asarray(ep_seed, dtype=np.int64),
            "outcomes": outcomes, "steps_per_episode": steps_per_episode}


def relabel_arrays(visited: dict) -> dict:
    """Episode-level split of the relabeled states (same 80/20 rule, relabel namespace)."""
    flags = V.split_table(V.RELABEL_SEED_START, V.RELABEL_EPISODES, extra=1)
    is_train = flags[visited["ep_seed"] - V.RELABEL_SEED_START]
    return {"train": (visited["obs"][is_train], visited["act"][is_train]), "val": (visited["obs"][~is_train], visited["act"][~is_train])}


def append_datasets(base: dict, extra: dict) -> dict:
    return {k: (np.concatenate([base[k][0], extra[k][0]]), np.concatenate([base[k][1], extra[k][1]])) for k in ("train", "val")}
