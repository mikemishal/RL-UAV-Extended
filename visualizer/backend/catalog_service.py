from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .application_error import ApplicationError
from .config import Settings, VISUALIZER_ROOT

logger = logging.getLogger(__name__)
CATALOG_TTL_SECONDS = 30.0
CATALOG_TIMEOUT_SECONDS = 180.0


class CatalogProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._catalog: dict[str, object] | None = None
        self._loaded_at = 0.0

    def get(self) -> dict[str, object]:
        with self._lock:
            if self._catalog is not None and time.monotonic() - self._loaded_at < CATALOG_TTL_SECONDS:
                return copy.deepcopy(self._catalog)
            catalog = self._load()
            self._catalog = catalog
            self._loaded_at = time.monotonic()
            return copy.deepcopy(catalog)

    def _load(self) -> dict[str, object]:
        env = os.environ.copy()
        if self.settings.research_data_root is None:
            env.pop("VISUALIZER_RESEARCH_DATA_ROOT", None)
        else:
            env["VISUALIZER_RESEARCH_DATA_ROOT"] = str(self.settings.research_data_root)
        command = [str(self.settings.simulation_python), "-m", "visualizer.backend.catalog_cli"]
        try:
            result = subprocess.run(
                command,
                cwd=VISUALIZER_ROOT.parent,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=CATALOG_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ApplicationError("CATALOG_TIMEOUT", "Policy catalog inspection exceeded its 180-second limit.", 504) from exc
        except OSError as exc:
            logger.error("Cannot start catalog worker", exc_info=exc)
            raise ApplicationError("SIMULATION_RUNTIME_UNAVAILABLE", "Cannot start the configured simulation Python runtime.", 503) from exc
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            logger.error("Catalog worker returned invalid output (exit %s): %s", result.returncode, result.stderr[-4000:])
            raise ApplicationError(
                "CATALOG_WORKER_FAILED",
                "Policy catalog inspection failed. Check that the configured simulation interpreter has the research dependencies.",
                503,
            ) from exc
        if result.returncode != 0:
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str):
                raise ApplicationError(error["code"], error["message"], 503)
            logger.error("Catalog worker failed (exit %s): %s", result.returncode, result.stderr[-4000:])
            raise ApplicationError("CATALOG_WORKER_FAILED", "Policy catalog inspection failed.", 503)
        if not isinstance(payload, dict) or not isinstance(payload.get("defenders"), list) \
                or not isinstance(payload.get("attackers"), list) or not isinstance(payload.get("scenarios"), list):
            raise ApplicationError("INVALID_CATALOG", "The catalog worker returned an invalid catalog.", 500)
        return payload
