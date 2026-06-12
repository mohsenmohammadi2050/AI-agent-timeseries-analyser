from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(
        title="Agentic Time Series Analysis API",
        version="0.1.0",
        description="Manager-facing API for agentic time series and prediction analysis.",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def chat_interface() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
