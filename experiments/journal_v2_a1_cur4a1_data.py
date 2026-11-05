"""Expert dataset for curriculum Stage 4A-1: canonical A0 episodes on TRANSLATED stationary-asset scenarios -> (37-D observation, normalized A0 direction) pairs.

Restart-safe: each shard is generated under an exclusive lock, written atomically (npz, then its hashed sidecar manifest LAST); a verified shard is never regenerated. `dataset_manifest.json`
(written last) records every shard hash, the dataset hash, the split definition and the translation/geometry coverage (asset-position distribution, attacker-position distribution, relative
range/bearing, heading error, altitude, quadrant/diagonal coverage).
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur4a1 as Z
from experiments import journal_v2_a1_eg_state as P

DATA_DIR = Z.RESULTS_ROOT / "dataset"
SHARD_EPISODES = Z.EXPERT_EPISODES // Z.EXPERT_SHARDS
SCHEMA = "a1_cur4a1_expert_shard/1"


def shard_path(i: int) -> Path:
    return DATA_DIR / f"expert_shard_{i:02d}.npz"


def shard_seeds(i: int) -> range:
    start = Z.EXPERT_SEED_START + i * SHARD_EPISODES
    return range(start, start + SHARD_EPISODES)


def verify_shard(i: int, data_dir: Path | None = None) -> dict | None:
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    try:
        side = json.loads((base / f"expert_shard_{i:02d}.json").read_text())
        if side.get("schema") != SCHEMA or side.get("shard") != i or side.get("env_config_sha256") != Z.config_sha256():
            return None
        if P.sha256_file(base / f"expert_shard_{i:02d}.npz") != side["npz_sha256"]:
            return None
        return side
    except (OSError, ValueError, KeyError, TypeError):
        return None


def generate_expert_episodes(seeds) -> dict:
    """A0 (scripted, no override) on translated `seeds`; at every decision the masked observation and the A0 label of the same (translated) state are recorded, plus translation/geometry metadata."""
    env = Z.TranslatedStage1V2Env(0, 0, 0, scripted_a0=True)
    obs_rows, act_rows, ep_idx, step_idx = [], [], [], []
    meta = {"seed": [], "steps": [], "initial_range": [], "bearing_deg": [], "altitude": [], "heading_offset_deg": [], "caught": [], "asset_x": [], "asset_y": [], "translation_x": [], "translation_y": []}
    for s in seeds:
        obs, _ = env.reset(options={"scenario": int(s)})
        e = env.soldier_env
        rel = np.asarray(e._enemy_pos, dtype=np.float64) - np.asarray(e._soldier_pos, dtype=np.float64)
        done, t = False, 0
        while not done:
            obs_rows.append(obs.astype(np.float32))
            act_rows.append(env.expert_action())
            ep_idx.append(int(s))
            step_idx.append(t)
            obs, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
            done, t = te or tr, t + 1
        ep = info["a1_episode"]
        meta["seed"].append(int(s)); meta["steps"].append(int(ep["steps"])); meta["initial_range"].append(float(ep["initial_range_m"]))
        meta["bearing_deg"].append(float(np.degrees(np.arctan2(rel[1], rel[0])))); meta["altitude"].append(float(ep["initial_altitude_m"]))
        meta["heading_offset_deg"].append(float(ep["initial_heading_offset_deg"])); meta["caught"].append(bool(ep["outcome"] == "soldier_caught"))
        meta["asset_x"].append(float(ep["asset_position_m"][0])); meta["asset_y"].append(float(ep["asset_position_m"][1]))
        meta["translation_x"].append(float(ep["translation_m"][0])); meta["translation_y"].append(float(ep["translation_m"][1]))
    return {"obs": np.asarray(obs_rows, dtype=np.float32), "act": np.asarray(act_rows, dtype=np.float32), "ep_seed": np.asarray(ep_idx, dtype=np.int64),
            "step": np.asarray(step_idx, dtype=np.int16), **{f"meta_{k}": np.asarray(v) for k, v in meta.items()}}


def npz_bytes(arrays: dict) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def write_shard(i: int, data_dir: Path | None = None, seeds=None) -> dict:
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    if verify_shard(i, base) is not None:
        return {"shard": i, "skipped": True}
    lock = P.CellLock(base / f"expert_shard_{i:02d}")
    if not lock.acquire():
        return {"shard": i, "skipped": True, "reason": "another worker owns this shard"}
    try:
        started = time.time()
        seeds = list(seeds if seeds is not None else shard_seeds(i))
        arrays = generate_expert_episodes(seeds)
        payload = npz_bytes(arrays)
        P.atomic_write_bytes(base / f"expert_shard_{i:02d}.npz", payload)
        side = {"schema": SCHEMA, "shard": i, "episodes": len(seeds), "seed_range": [seeds[0], seeds[-1]], "pairs": int(arrays["obs"].shape[0]), "npz_sha256": hashlib.sha256(payload).hexdigest(),
                "env_config_sha256": Z.config_sha256(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        P.atomic_write_json(base / f"expert_shard_{i:02d}.json", side)       # sidecar LAST = commit marker
        return {"shard": i, "pairs": side["pairs"], "seconds": round(time.time() - started, 1)}
    finally:
        lock.release()


def _worker(i: int) -> dict:
    return write_shard(i)


def generate(workers: int = 10) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DATA_DIR.glob(".*.tmp*"):
        stale.unlink(missing_ok=True)
    pending = [i for i in range(Z.EXPERT_SHARDS) if verify_shard(i) is None]
    print(f"{Z.EXPERT_SHARDS - len(pending)} shards already verified, {len(pending)} to generate", flush=True)
    if pending:
        with mp.get_context("spawn").Pool(min(workers, len(pending))) as pool:
            for r in pool.imap_unordered(_worker, pending):
                print(r, flush=True)
    return finalize()


# ----------------------------------------------------------------------------- loading, split, coverage, manifest
def load_all(data_dir: Path | None = None) -> dict:
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    parts = []
    for i in range(Z.EXPERT_SHARDS):
        if verify_shard(i, base) is None:
            raise RuntimeError(f"shard {i} is missing or fails verification")
        with np.load(base / f"expert_shard_{i:02d}.npz") as zf:
            parts.append({k: zf[k] for k in zf.files})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def split(data: dict) -> dict:
    """Episode-level 80/20 split (no episode leakage)."""
    train_flags = V1V2.split_table(Z.EXPERT_SEED_START, Z.EXPERT_EPISODES, Z.TRAIN_FRACTION)
    index = data["ep_seed"] - Z.EXPERT_SEED_START
    is_train = train_flags[index]
    seeds = np.unique(data["ep_seed"])
    tr = {int(s) for s in seeds if train_flags[int(s) - Z.EXPERT_SEED_START]}
    va = {int(s) for s in seeds} - tr
    return {"train": (data["obs"][is_train], data["act"][is_train]), "val": (data["obs"][~is_train], data["act"][~is_train]), "train_episodes": tr, "val_episodes": va}


def coverage(data: dict) -> dict:
    steps, rng, bearing, alt, off = data["meta_steps"], data["meta_initial_range"], data["meta_bearing_deg"], data["meta_altitude"], data["meta_heading_offset_deg"]
    ax, ay = data["meta_asset_x"], data["meta_asset_y"]
    quadrants = {"+x+y": int(((ax >= 0) & (ay >= 0)).sum()), "+x-y": int(((ax >= 0) & (ay < 0)).sum()), "-x+y": int(((ax < 0) & (ay >= 0)).sum()), "-x-y": int(((ax < 0) & (ay < 0)).sum())}
    diagonal = int((np.abs(np.abs(ax) - np.abs(ay)) <= 0.25 * np.maximum(np.abs(ax), np.abs(ay)).clip(min=1.0)).sum())
    steps_of = dict(zip(data["meta_seed"].tolist(), steps.tolist()))
    total_steps = np.array([steps_of[int(s)] for s in data["ep_seed"]])
    late = int((total_steps - data["step"] <= 2).sum())
    return {
        "episodes": int(len(steps)), "pairs": int(data["obs"].shape[0]), "pairs_per_episode_mean": float(steps.mean()), "pairs_per_episode_min_max": [int(steps.min()), int(steps.max())],
        "caught_fraction": float(np.mean(data["meta_caught"])), "asset_position_m": {"x": [float(ax.min()), float(np.median(ax)), float(ax.max())], "y": [float(ay.min()), float(np.median(ay)), float(ay.max())]},
        "asset_quadrant_episode_counts": quadrants, "near_diagonal_episode_count": diagonal, "initial_range_m": [float(rng.min()), float(np.median(rng)), float(rng.max())],
        "initial_altitude_m": [float(alt.min()), float(np.median(alt)), float(alt.max())], "heading_offset_deg": [float(off.min()), float(np.median(off)), float(off.max())],
        "central_fraction": float(np.mean(np.maximum(np.abs(ax), np.abs(ay)) <= 10.0)), "outer_fraction": float(np.mean(np.maximum(np.abs(ax), np.abs(ay)) >= 30.0)),
        "late_approach_decisions_within_2_of_end": late,
    }


def finalize(data_dir: Path | None = None) -> dict:
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    data = load_all(base)
    sp = split(data)
    cov = coverage(data)
    manifest = {"schema": "a1_cur4a1_dataset_manifest/1", "episodes": Z.EXPERT_EPISODES, "pairs": int(data["obs"].shape[0]), "train_pairs": int(sp["train"][0].shape[0]),
                "val_pairs": int(sp["val"][0].shape[0]), "train_episodes": len(sp["train_episodes"]), "val_episodes": len(sp["val_episodes"]), "split_fraction": Z.TRAIN_FRACTION,
                "env_config_sha256": Z.config_sha256(), "dataset_sha256": hashlib.sha256(data["obs"].tobytes() + data["act"].tobytes()).hexdigest(), "coverage": cov,
                "shard_hashes": {i: verify_shard(i, base)["npz_sha256"] for i in range(Z.EXPERT_SHARDS)}, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    P.atomic_write_json(base / "dataset_manifest.json", manifest)
    return manifest
