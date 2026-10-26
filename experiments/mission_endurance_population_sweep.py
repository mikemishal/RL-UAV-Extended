"""Phase 8: passive-A0 400-episode sweep (C0-C3 x seeds 130000-130099) per endurance factor.

No learned policy, no training. Legacy asset motion (canonical default) throughout.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from experiments.journal_v2_avoidance_recorder import record, plain

OUT = Path("results/mission_endurance_study")
SETTINGS = {"disabled": None, "k3": 3.0, "k4": 4.0, "k5": 5.0}


def run(label: str) -> None:
    k = SETTINGS[label]
    directory = OUT / label
    directory.mkdir(parents=True, exist_ok=True)

    def transform(cfg):
        if k is None:
            return cfg
        return dataclasses.replace(cfg, enemy_endurance_enabled=True, enemy_endurance_path_factor=k)

    for level in ("C0", "C1", "C2", "C3"):
        for seed in range(130000, 130100):
            rec = record(level, seed, cfg_transform=transform)
            (directory / f"{level}_{seed}.json").write_text(json.dumps(rec, default=plain), encoding="utf-8")
        print(label, level, "done", flush=True)


if __name__ == "__main__":
    for label in (sys.argv[1:] or list(SETTINGS)):
        run(label)
