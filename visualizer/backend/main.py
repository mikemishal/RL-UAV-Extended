import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings, load_settings, local_output_root
from .errors import ErrorDetail, register_error_handlers

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    service: Literal["counter-uas-visualizer"] = "counter-uas-visualizer"
    phase: Literal[1] = 1
    status: Literal["ok", "degraded"]
    simulation_enabled: Literal[False] = False
    data_root: Literal["configured", "not_configured"]
    warnings: list[ErrorDetail]


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configured = settings if settings is not None else load_settings()
        local_output_root()
        application.state.settings = configured
        if configured.research_data_root is None:
            logger.warning("DATA_ROOT_NOT_CONFIGURED: set VISUALIZER_RESEARCH_DATA_ROOT in visualizer\\.env")
        yield

    application = FastAPI(title="Counter-UAS Simulation Explorer", version="0.1.0", lifespan=lifespan)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    register_error_handlers(application)

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        configured = application.state.settings.research_data_root is not None
        warnings = [] if configured else [
            ErrorDetail(code="DATA_ROOT_NOT_CONFIGURED", message="Set VISUALIZER_RESEARCH_DATA_ROOT in the local backend configuration.")
        ]
        return HealthResponse(
            status="ok" if configured else "degraded",
            data_root="configured" if configured else "not_configured",
            warnings=warnings,
        )

    return application


app = create_app()
