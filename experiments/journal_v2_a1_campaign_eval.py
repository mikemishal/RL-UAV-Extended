"""Pre-registered V2 A1 evaluation (seeds 140000-140099) with full per-step forensic traces.

    python -m experiments.journal_v2_a1_campaign_eval evaluate [--workers N]
    python -m experiments.journal_v2_a1_campaign_eval passive          # small passive diagnostic set (visualization only)

Episode records come from the frozen `journal_v2_a1_evaluation.run_episode`; this module only (a) wraps the
environment with a read-only trace recorder, (b) fans the 80 (attacker x level x defender) tasks over processes and
(c) checks the integrity of the 8,000-episode result. Nothing here changes the environment, reward or protocol.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

import experiments.journal_v2_a1_training as T
from experiments import journal_v2_a1 as A
from experiments.journal_v2_a1_evaluation import run_episode

EVAL_ROOT = T.A1_V2_RESULTS_ROOT / "evaluation"
TASK_DIR = EVAL_ROOT / "tasks"
ATTACKERS = ("A0", "A1-201", "A1-202", "A1-203")
SEEDS = tuple(A.evaluation_seeds())
FAMILIES = A.A1_V2_DEFENDER_FAMILIES
LEVELS = A.A1_V2_CLUTTER_LEVELS


class TraceEnv:
    """Read-only recorder around `A1V2TrainingEnv`; exposes exactly what `run_episode` uses."""

    def __init__(self, env: A.A1V2TrainingEnv):
        self._env = env
        self._steps: list[dict] = []
        self._obs = None

    @property
    def soldier_env(self):
        return self._env.soldier_env

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        self._obs, self._steps = obs, []
        return obs, info

    def step(self, action):
        obs_before = self._obs
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._steps.append({
            "action": np.asarray(action, dtype=np.float32).copy(), "obs_before": np.asarray(obs_before, dtype=np.float32).copy(),
            "enemy_pos": np.asarray(info["enemy_pos"], dtype=np.float32), "asset_pos": np.asarray(info["soldier_pos"], dtype=np.float32),
            "defender_pos": np.asarray(info["defender_pos"], dtype=np.float32),
            "avoid_active": bool(info.get("enemy_obstacle_avoidance_active")), "avoid_mode": str(info.get("enemy_obstacle_avoidance_mode")),
            "final_cmd": np.asarray(info["enemy_final_command"], dtype=np.float32), "mission_cmd": np.asarray(info["enemy_mission_command"], dtype=np.float32),
            "actual_min_clearance": float(info["enemy_actual_min_clearance"]), "min_obstacle_clearance": float(info["enemy_min_obstacle_clearance"]),
            "n_perceived_static": len(info.get("enemy_perceived_static_ids", ())), "n_perceived_dynamic": len(info.get("enemy_perceived_dynamic_ids", ())),
            "perceived_static_ids": tuple(int(i) for i in info.get("enemy_perceived_static_ids", ())),
            "perceived_dynamic_ids": tuple(int(i) for i in info.get("enemy_perceived_dynamic_ids", ())),
            "range_to_asset": float(info["enemy_soldier_dist"]), "range_to_defender": float(info["defender_enemy_dist"]),
            "endurance_remaining_m": float(info["enemy_endurance_remaining_m"]), "reward": float(reward),
            "collision_type": info.get("enemy_collision_event_type"), "collision_obstacle_id": info.get("enemy_collision_obstacle_id"),
            "collision_point": None if info.get("enemy_collision_event_point") is None else np.asarray(info["enemy_collision_event_point"], dtype=np.float32),
            "outcome": info["outcome"],
        })
        self._obs = obs
        return obs, reward, terminated, truncated, info

    def trace(self) -> list[dict]:
        return self._steps


def _load_attacker(attacker_id: str):
    if attacker_id == "A0":
        return None
    from stable_baselines3 import PPO

    registry = json.loads(T.A1_V2_RESULTS_ROOT.joinpath("a1_v2_model_registry.json").read_text())
    entry = registry["attackers"][f"{attacker_id}-V2"]
    path = T.PROJECT_ROOT / entry["path"]
    assert T.sha256_file(path) == entry["sha256"], f"{attacker_id}: model hash differs from the registry"
    model = PPO.load(str(path.with_suffix("")), device="cpu")
    assert model.observation_space.shape == (37,) and model.action_space.shape == (3,)
    return model


def run_task(task: tuple) -> dict:
    attacker_id, level, family, seeds, out_dir = task
    out = Path(out_dir) / f"{attacker_id}__{level}__{family}.pkl"
    if out.exists():
        return {"task": task[:3], "skipped": True}
    import torch

    torch.set_num_threads(1)
    model = _load_attacker(attacker_id)
    env = TraceEnv(A.A1V2TrainingEnv(0, 0, defender_bank=A.A1V2DefenderBank(), scripted_a0=model is None))
    records, traces = [], {}
    t0 = time.time()
    for seed in seeds:
        record = run_episode(model, attacker_id, level, family, int(seed), env=env)
        records.append(record)
        traces[int(seed)] = env.trace()
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump({"records": records, "traces": traces}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    return {"task": task[:3], "episodes": len(records), "seconds": time.time() - t0}


def evaluate(workers: int = 18, attackers=ATTACKERS, out_dir: Path = TASK_DIR, seeds=SEEDS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(a, lvl, fam, tuple(seeds), str(out_dir)) for a in attackers for lvl in LEVELS for fam in FAMILIES]
    tasks.sort(key=lambda t: (t[2] != "rmpc",))  # heaviest first for load balance
    print(f"{len(tasks)} tasks, {workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_task, tasks), 1):
            print(i, result, flush=True)


def load_all(out_dir: Path = TASK_DIR) -> tuple[list[dict], dict]:
    """Eager load (small runs / tests only). The full 8,000-episode evaluation holds ~5 GB of traces; use
    `scan_tasks` + `TraceStore` for it."""
    records, traces = [], {}
    for path in sorted(out_dir.glob("*.pkl")):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        records.extend(payload["records"])
        for record in payload["records"]:
            traces[(record["attacker"], record["level"], record["defender"], record["seed"])] = payload["traces"][record["seed"]]
    return records, traces


def scan_tasks(derive, out_dir: Path = TASK_DIR) -> list[dict]:
    """Stream the task files one at a time; `derive(record, trace) -> dict` adds per-episode scalars to each record
    so the (large) traces can be discarded."""
    records = []
    for path in sorted(out_dir.glob("*.pkl")):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        for record in payload["records"]:
            records.append({**record, **derive(record, payload["traces"][record["seed"]])})
        del payload
    return records


class TraceStore:
    """Dict-like lazy access to single-episode traces (keeps only the most recently used task file in memory)."""

    def __init__(self, out_dir: Path = TASK_DIR):
        self.out_dir, self._name, self._payload = out_dir, None, None

    def __getitem__(self, key):
        attacker, level, defender, seed = key
        name = f"{attacker}__{level}__{defender}.pkl"
        if name != self._name:
            with open(self.out_dir / name, "rb") as handle:
                self._payload = pickle.load(handle)
            self._name = name
        return self._payload["traces"][int(seed)]


def integrity_check(records: list[dict]) -> dict:
    """8,000 unique episodes, identical scenario sets across attackers, matched initial conditions."""
    keys = {(r["attacker"], r["level"], r["defender"], r["seed"]) for r in records}
    assert len(keys) == len(records) == len(ATTACKERS) * len(LEVELS) * len(FAMILIES) * len(SEEDS), (len(keys), len(records))
    by_scenario: dict = {}
    for r in records:
        by_scenario.setdefault((r["level"], r["defender"], r["seed"]), {})[r["attacker"]] = r
    assert len(by_scenario) == len(LEVELS) * len(FAMILIES) * len(SEEDS)
    max_range_gap = 0.0
    for scenario, group in by_scenario.items():
        assert set(group) == set(ATTACKERS), scenario
        ranges = [g["initial_range_m"] for g in group.values()]
        max_range_gap = max(max_range_gap, max(ranges) - min(ranges))
    assert max_range_gap < 1e-6, f"matched scenarios differ in initial attacker-asset range by {max_range_gap}"
    assert all(r["outcome"] != "ongoing" for r in records)
    assert min(SEEDS) == 140_000 and max(SEEDS) == 140_099
    assert "passive" not in {r["defender"] for r in records}
    return {"episodes": len(records), "unique": len(keys), "matched_scenarios": len(by_scenario),
            "max_initial_range_gap_m": max_range_gap, "seed_range": [min(SEEDS), max(SEEDS)], "attackers": list(ATTACKERS)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("evaluate",))
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    evaluate(args.workers)
    print("evaluation tasks complete; run experiments.journal_v2_a1_campaign_analysis for integrity + analysis", flush=True)
