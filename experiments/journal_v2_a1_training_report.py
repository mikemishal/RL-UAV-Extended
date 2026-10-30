"""Training-health report, learning curves and anomaly checks for the V2 A1 campaign (read-only on all results)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.journal_v2_a1_training as T  # noqa: E402
from experiments import journal_v2_a1 as A  # noqa: E402

REPORT_DIR = T.A1_V2_RESULTS_ROOT / "training_report"
BIN = 20_000
COLLISIONS = ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision")
INTERCEPTS = ("intercepted", "unsafe_intercept")
COLORS = {201: "#1f77b4", 202: "#d62728", 203: "#2ca02c"}


def episodes(seed: int) -> pd.DataFrame:
    path = T.run_paths(seed).logs / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # A resumed run restarts from its last periodic checkpoint; episodes logged after that checkpoint by the
    # interrupted attempt are superseded by the resumed attempt and are dropped (timesteps move backwards).
    for i in range(1, len(rows)):
        if rows[i]["timesteps"] < rows[i - 1]["timesteps"]:
            checkpoint = (rows[i]["timesteps"] // A.A1_V2_CHECKPOINT_FREQ) * A.A1_V2_CHECKPOINT_FREQ
            rows = [r for j, r in enumerate(rows) if j >= i or r["timesteps"] <= checkpoint]
            break
    frame = pd.DataFrame(rows)
    for key in ("terminal", "progress", "path", "time"):
        frame[key] = frame["components"].map(lambda c, k=key: c[k])
    frame["captured"] = frame.outcome == "soldier_caught"
    frame["intercepted"] = frame.outcome.isin(INTERCEPTS)
    frame["collided"] = frame.outcome.isin(COLLISIONS)
    frame["exhausted"] = frame.outcome == "enemy_endurance_exhausted"
    frame["timed_out"] = frame.outcome == "timeout"
    frame["bin"] = (frame.timesteps // BIN).astype(int)
    return frame


def aggregate(frame: pd.DataFrame, by: str = "bin") -> pd.DataFrame:
    def stats(g: pd.DataFrame) -> pd.Series:
        pe = g.path_excess.replace([np.inf, -np.inf], np.nan)
        cap = g[g.captured]
        return pd.Series({
            "timesteps_end": g.timesteps.max(), "episodes": len(g), "capture_rate": g.captured.mean(),
            "interception_rate": g.intercepted.mean(), "collision_rate": g.collided.mean(), "exhaustion_rate": g.exhausted.mean(),
            "timeout_rate": g.timed_out.mean(), "mean_return": g.return_undiscounted.mean(), "terminal": g.terminal.mean(),
            "progress": g.progress.mean(), "path": g.path.mean(), "time": g.time.mean(), "mean_path_excess": pe.mean(),
            "gt1.3x": int((pe > 1.3).sum()), "gt2x": int((pe > 2).sum()), "gt5x": int((pe > 5).sum()),
            "captured_gt1.3x": int((cap.path_excess > 1.3).sum()), "captured_gt2x": int((cap.path_excess > 2).sum()),
            "captured_gt5x": int((cap.path_excess > 5).sum()),
        })
    return frame.groupby(by).apply(stats, include_groups=False).reset_index()


def anomaly_checks(seed: int, frame: pd.DataFrame) -> dict:
    import torch
    from stable_baselines3 import PPO

    paths = T.run_paths(seed)
    checks: dict = {}
    checks["terminal_values_only_pm10"] = bool(set(frame.terminal.round(9).unique()) <= {10.0, -10.0})
    checks["reward_components_finite"] = bool(np.isfinite(frame[["terminal", "progress", "path", "time", "return_undiscounted"]].to_numpy()).all())
    checks["progress_range"] = [float(frame.progress.min()), float(frame.progress.max())]
    checks["path_range"] = [float(frame.path.min()), float(frame.path.max())]
    checks["time_range"] = [float(frame.time.min()), float(frame.time.max())]
    checks["return_range"] = [float(frame.return_undiscounted.min()), float(frame.return_undiscounted.max())]
    checks["path_penalty_within_budget"] = bool(frame.path.min() >= -1.0 - 1e-6)
    checks["unknown_outcomes"] = sorted(set(frame.outcome) - {"soldier_caught", *INTERCEPTS, *COLLISIONS, "enemy_endurance_exhausted", "timeout",
                                                               "defender_collision"})
    models = sorted(paths.checkpoints.glob("*.zip"))
    probe = np.random.default_rng(0).uniform(-1, 1, (256, 37)).astype(np.float32)
    probe[:, 36] = np.random.default_rng(1).uniform(0, 1, 256)
    norms, model_report = [], {}
    for model_path in models:
        if model_path.name in ("resume_latest.zip",):
            continue
        model = PPO.load(str(model_path.with_suffix("")), device="cpu")
        finite = all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
        actions, _ = model.predict(probe, deterministic=True)
        norm = float(math.sqrt(sum(float((p.detach() ** 2).sum()) for p in model.policy.parameters())))
        model_report[model_path.name] = {"params_finite": finite, "actions_finite": bool(np.isfinite(actions).all()),
                                         "actions_within_box": bool((np.abs(actions) <= 1.0 + 1e-6).all()), "param_l2": round(norm, 3),
                                         "num_timesteps": int(model.num_timesteps), "device": str(model.device)}
        norms.append(norm)
    checks["models"] = model_report
    checks["all_models_finite_and_valid"] = all(v["params_finite"] and v["actions_finite"] and v["actions_within_box"] for v in model_report.values())
    checks["param_norm_max_over_min"] = float(max(norms) / min(norms)) if norms else None
    checks["final_model_exists"] = paths.final_model.exists()
    sidecar = json.loads(paths.final_model.with_suffix(".manifest.json").read_text())
    checks["final_manifest_sha_matches"] = sidecar["model_sha256"] == T.sha256_file(paths.final_model)
    checks["final_num_timesteps"] = sidecar["num_timesteps"]
    checks["periodic_checkpoints_each_with_manifest"] = all(m.with_suffix(".manifest.json").exists() for m in models if m.name.startswith("a1_v2_seed"))
    return checks


def plots(frames: dict[int, pd.DataFrame]) -> list[str]:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    metrics = [("capture_rate", "Asset-capture rate"), ("collision_rate", "Obstacle-collision rate"),
               ("exhaustion_rate", "Endurance-exhaustion rate"), ("mean_return", "Mean episode return"),
               ("mean_path_excess", "Mean path-excess (path / chord)")]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    aggs = {seed: aggregate(frame) for seed, frame in frames.items()}
    written = []
    combined, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (key, label) in zip(axes.flat, metrics):
        single, single_ax = plt.subplots(figsize=(4.6, 3.2))
        for seed, agg in aggs.items():
            for target in (ax, single_ax):
                target.plot(agg.timesteps_end, agg[key], color=COLORS[seed], lw=1.4, label=f"A1-{seed}")
        for target in (ax, single_ax):
            target.set_xlabel("PPO timesteps")
            target.set_ylabel(label)
        single_ax.legend(frameon=False)
        single.tight_layout()
        for ext in ("png", "pdf"):
            path = REPORT_DIR / f"curve_{key}.{ext}"
            single.savefig(path, dpi=200 if ext == "png" else None)
            written.append(str(path))
        plt.close(single)
    last = axes.flat[-1]
    for seed, agg in aggs.items():
        last.plot(agg.timesteps_end, agg.interception_rate, color=COLORS[seed], lw=1.4, label=f"A1-{seed}")
    last.set_xlabel("PPO timesteps")
    last.set_ylabel("Defender-interception rate")
    axes.flat[0].legend(frameon=False)
    combined.tight_layout()
    for ext in ("png", "pdf"):
        path = REPORT_DIR / f"learning_curves.{ext}"
        combined.savefig(path, dpi=200 if ext == "png" else None)
        written.append(str(path))
    plt.close(combined)
    return written


def build_report(seeds=A.A1_V2_PPO_SEEDS) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {seed: episodes(seed) for seed in seeds}
    report: dict = {"seeds": {}}
    for seed, frame in frames.items():
        paths = T.run_paths(seed)
        run = json.loads(paths.run_manifest.read_text())
        final_100k = frame[frame.timesteps > frame.timesteps.max() - 100_000]
        first_100k = frame[frame.timesteps <= 100_000]
        validation = [json.loads(line) for line in paths.summaries.joinpath("validation.jsonl").read_text().splitlines()]
        agg = aggregate(frame)
        agg.to_csv(REPORT_DIR / f"seed_{seed}_binned_training_curve.csv", index=False)
        report["seeds"][seed] = {
            "status": run.get("status"), "num_timesteps": run.get("num_timesteps"), "episodes": int(len(frame)),
            "episode_steps_sum": int(frame.steps.sum()), "overall": aggregate(frame.assign(g=0), "g").iloc[0].to_dict(),
            "first_100k": aggregate(first_100k.assign(g=0), "g").iloc[0].to_dict(),
            "last_100k": aggregate(final_100k.assign(g=0), "g").iloc[0].to_dict(),
            "validation": [{"timesteps": v["timesteps"], **{k: v[k] for k in ("capture_rate", "interception_rate", "obstacle_collision_rate",
                                                                               "endurance_exhaustion_rate", "timeout_rate", "mean_return")}}
                           for v in validation],
            "anomaly_checks": anomaly_checks(seed, frame),
        }
    report["figures"] = plots(frames)
    (REPORT_DIR / "training_health_report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    out = build_report()
    for seed, r in out["seeds"].items():
        print(seed, r["status"], r["num_timesteps"], r["episodes"], {k: round(v, 3) for k, v in r["last_100k"].items() if k.endswith("rate") or k == "mean_return"})
