from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.logging_config import setup_logging, log
from app.api.routes import router, set_estimator
from app.services.depth_estimator import DepthEstimator


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("DepthWizard starting up...")

    estimator = DepthEstimator()
    set_estimator(estimator)

    log.info("DepthWizard ready!")
    yield
    log.info("DepthWizard shutting down.")


app = FastAPI(
    title="DepthWizard API",
    description="Single-View Height Estimation & 3D Flythrough — SIH26175",
    version="0.1.0-mvp",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "service": "DepthWizard API",
        "project": "SIH26175 - ISRO Space Applications Centre",
        "description": "Single-View Satellite Elevation Estimation & 3D Flythrough",
        "status": "online",
        "docs_url": "/docs",
        "api_health": "/api/health",
        "api_upload": "/api/upload",
        "device": device,
        "version": "1.0.0"
    }
