"""Post-hoc diagnostics of the Stage-1 policies (NOT part of the mastery protocol; explains WHAT the policies do). Deterministic policy, validation set A.

Per seed (final/accepted model): outcome distribution, mean cosine between the executed direction and the line of sight to the asset, share of steps spent at the arena
boundary, share of steps with closing rate > 1 m/s, mean action norm, learned action std; plus training-side statistics (episodes and episode length per rollout).
"""

from __future__ import annotations

import json

import numpy as np

from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1_state as TS
from experiments import journal_v2_a1_cur1_training as TR

OUT = TS.STATE_DIR / "diagnostics.json"


def rollout_policy(model, seeds):
    env = C.Stage1Env(0, 0, 0)
    rows = []
    for s in seeds:
        obs, _ = env.reset(options={"scenario": int(s)})
        done, cos, boundary, closing, norms, steps = False, [], 0, 0, [], 0
        prev = env._range
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            u, _held = C.desired_direction(action, env.soldier_env._enemy_vel)
            e = env.soldier_env
            los = np.asarray(e._soldier_pos, dtype=float) - np.asarray(e._enemy_pos, dtype=float)
            cos.append(float(np.dot(u, los / np.linalg.norm(los))))
            norms.append(float(np.linalg.norm(np.clip(action, -1, 1))))
            obs, _r, te, tr, info = env.step(action)
            done = te or tr
            steps += 1
            pos = np.asarray(info["enemy_pos"], dtype=float)
            boundary += int(abs(pos[0]) >= 49.9 or abs(pos[1]) >= 49.9)
            closing += int((prev - env._range) / env._dt > 1.0)
            prev = env._range
        ep = info["a1_episode"]
        rows.append({"outcome": ep["outcome"], "steps": steps, "mean_cos_to_los": float(np.mean(cos)), "boundary_share": boundary / steps, "closing_share": closing / steps,
                     "mean_action_norm": float(np.mean(norms)), "terminal_distance_m": ep["terminal_distance_m"], "initial_range_m": ep["initial_range_m"],
                     "initial_heading_offset_deg": ep["initial_heading_offset_deg"]})
    return rows


def _diagnose_seed(item):
    from stable_baselines3 import PPO
    import collections

    seed, m = item
    layout = TR.make_layout(seed, TR.select_scratch_root())
    model = PPO.load(str(layout.accepted_model).removesuffix(".zip"), device="cpu")
    rows = rollout_policy(model, list(C.validation_seeds("A"))[:40])
    return str(seed), {
        "model_rollout": m["accepted_rollout"], "episodes": len(rows), "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
        "mean_cos_to_los": float(np.mean([r["mean_cos_to_los"] for r in rows])), "mean_boundary_share": float(np.mean([r["boundary_share"] for r in rows])),
        "mean_closing_share": float(np.mean([r["closing_share"] for r in rows])), "mean_action_norm": float(np.mean([r["mean_action_norm"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])), "action_std_mean": float(np.exp(model.policy.log_std.detach().numpy()).mean()),
        "action_std": [float(x) for x in np.exp(model.policy.log_std.detach().numpy())],
        "heading_offset_deg": {"captured_mean": float(np.mean([r["initial_heading_offset_deg"] for r in rows if r["outcome"] == "soldier_caught"] or [float("nan")])),
                               "failed_mean": float(np.mean([r["initial_heading_offset_deg"] for r in rows if r["outcome"] != "soldier_caught"] or [float("nan")]))},
        "training_per_rollout": [{"rollout": r["rollout"], "episodes": r["training_summary"].get("episodes"), "capture_rate": r["training_summary"].get("capture_rate"),
                                  "mean_duration_s": r["training_summary"].get("mean_duration_s"), "mean_return": r["training_summary"].get("mean_return")} for r in m["rollouts"]],
    }


def diagnose(manifests: dict) -> dict:
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(len(manifests)) as pool:
        out = dict(pool.map(_diagnose_seed, list(manifests.items())))
    TS.atomic_write_json(OUT, out)
    return out


if __name__ == "__main__":
    from experiments.journal_v2_a1_cur1_eval import seed_manifests

    res = diagnose(seed_manifests())
    for s, v in res.items():
        print(s, v["outcomes"], "cos", round(v["mean_cos_to_los"], 3), "boundary", round(v["mean_boundary_share"], 3), "closing", round(v["mean_closing_share"], 3),
              "std", round(v["action_std_mean"], 3), "steps", round(v["mean_steps"], 1))
