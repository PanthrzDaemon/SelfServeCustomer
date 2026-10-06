from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.db.database import connect, init_schema


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Add tables introduced after the database was seeded (CREATE TABLE IF NOT EXISTS).
    if Path(settings.database_path).exists():
        with connect(settings.database_path) as conn:
            init_schema(conn)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "status": "running",
        "environment": settings.environment,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
