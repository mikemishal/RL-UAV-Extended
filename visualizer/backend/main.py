import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query, Response, status
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .catalog_service import CatalogProvider
from .config import Settings, load_settings, local_output_root
from .errors import ErrorDetail, register_error_handlers
from .replay_contract import ReplayBundle
from .replay_jobs import ReplayCatalog, ReplayJob, ReplayJobManager
from .replay_requests import ReplayBatchRequest

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    service: Literal["counter-uas-visualizer"] = "counter-uas-visualizer"
    phase: Literal[3] = 3
    status: Literal["ok", "degraded"]
    simulation_enabled: bool
    data_root: Literal["configured", "not_configured"]
    warnings: list[ErrorDetail]


def create_app(
    settings: Settings | None = None,
    *,
    catalog_provider: CatalogProvider | None = None,
    job_manager: ReplayJobManager | None = None,
) -> FastAPI:
    configured = settings if settings is not None else load_settings()
    catalog = catalog_provider if catalog_provider is not None else CatalogProvider(configured)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        local_output_root()
        application.state.settings = configured
        application.state.catalog = catalog
        application.state.jobs = job_manager if job_manager is not None else ReplayJobManager(configured, catalog)
        if configured.research_data_root is None:
            logger.warning("DATA_ROOT_NOT_CONFIGURED: set VISUALIZER_RESEARCH_DATA_ROOT in visualizer\\.env")
        try:
            yield
        finally:
            application.state.jobs.shutdown()

    application = FastAPI(title="Counter-UAS Simulation Explorer", version="0.1.0", lifespan=lifespan)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[configured.frontend_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    register_error_handlers(application)

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        configured = application.state.settings.research_data_root is not None
        warnings = [] if configured else [
            ErrorDetail(code="DATA_ROOT_NOT_CONFIGURED", message="Set VISUALIZER_RESEARCH_DATA_ROOT in the local backend configuration.")
        ]
        return HealthResponse(
            status="ok" if configured else "degraded",
            simulation_enabled=configured,
            data_root="configured" if configured else "not_configured",
            warnings=warnings,
        )

    @application.get("/api/policies")
    def policies() -> dict:
        snapshot = application.state.catalog.get()
        return {
            "source_commit": snapshot["source_commit"],
            "registry_sha256": snapshot["registry_sha256"],
            "deterministic_inference": True,
            "defenders": snapshot["defenders"],
            "attackers": snapshot["attackers"],
        }

    @application.get("/api/scenarios")
    def scenarios() -> dict:
        snapshot = application.state.catalog.get()
        return {"scenarios": snapshot["scenarios"]}

    @application.post("/api/replay-jobs", response_model=ReplayJob, status_code=status.HTTP_202_ACCEPTED)
    def create_replay_job(request: ReplayBatchRequest, response: Response) -> ReplayJob:
        job = application.state.jobs.start(request)
        response.headers["Location"] = f"/api/replay-jobs/{job.job_id}"
        return job

    @application.get("/api/replay-jobs/{job_id}", response_model=ReplayJob)
    def replay_job(job_id: str) -> ReplayJob:
        return application.state.jobs.get(job_id)

    @application.post("/api/replay-jobs/{job_id}/cancel", response_model=ReplayJob)
    def cancel_replay_job(job_id: str) -> ReplayJob:
        return application.state.jobs.cancel(job_id)

    @application.get("/api/replays", response_model=ReplayCatalog)
    def replays(limit: int = Query(default=100, ge=1, le=500)) -> ReplayCatalog:
        return application.state.jobs.list_replays(limit)

    @application.get("/api/replays/{replay_id}/summary")
    def replay_summary(replay_id: str) -> dict:
        bundle = application.state.jobs.get_replay(replay_id)
        episode = bundle.episodes[0]
        return {
            "replay_id": replay_id,
            "schema_version": bundle.schema_version,
            "simulation_seed": episode.simulation_seed,
            "defender": episode.defender.model_dump(mode="json"),
            "attacker": episode.attacker.model_dump(mode="json"),
            "environment": episode.environment.model_dump(mode="json"),
            "summary": episode.summary.model_dump(mode="json"),
            "provenance": bundle.provenance.model_dump(mode="json"),
        }

    @application.get("/api/replays/{replay_id}", response_model=ReplayBundle)
    def replay(replay_id: str) -> ReplayBundle:
        return application.state.jobs.get_replay(replay_id)

    return application


app = create_app()
