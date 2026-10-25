"""Phase 8: passive-A0 400-episode sweep (C0-C3 x seeds 130000-130099) per asset-motion setting.

No learned policy, no training. Reuses the existing recorder infrastructure (same decisions/
obstacle instrumentation as every prior round) with only the asset-motion config transformed.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from experiments.journal_v2_avoidance_recorder import record, plain

OUT = Path("results/asset_motion_model_study")
SETTINGS = {
    "legacy": ("legacy_gaussian_random_walk", 1.0),
    "tau0.5": ("gauss_markov_velocity", 0.5),
    "tau1.0": ("gauss_markov_velocity", 1.0),
    "tau2.0": ("gauss_markov_velocity", 2.0),
    "tau4.0": ("gauss_markov_velocity", 4.0),
}


def run(label: str) -> None:
    model, tau = SETTINGS[label]
    directory = OUT / label
    directory.mkdir(parents=True, exist_ok=True)

    def transform(cfg):
        return dataclasses.replace(cfg, asset_motion_model=model, asset_motion_tau_s=tau)

    for level in ("C0", "C1", "C2", "C3"):
        for seed in range(130000, 130100):
            rec = record(level, seed, cfg_transform=transform)
            (directory / f"{level}_{seed}.json").write_text(json.dumps(rec, default=plain), encoding="utf-8")
        print(label, level, "done", flush=True)


if __name__ == "__main__":
    for label in (sys.argv[1:] or list(SETTINGS)):
        run(label)
