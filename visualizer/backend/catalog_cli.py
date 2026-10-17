from __future__ import annotations

import json
import sys

from .application_error import ApplicationError
from .config import load_settings
from .replay_catalog import build_replay_catalog


def main() -> int:
    try:
        catalog = build_replay_catalog(load_settings())
    except ApplicationError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message}}))
        return 1
    print(json.dumps(catalog, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
