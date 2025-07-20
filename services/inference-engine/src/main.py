"""
VORTA Inference Engine - Main Application
Production-ready AI inference service with FastAPI, caching, and monitoring
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from vorta.api.dependencies import lifespan_context
from vorta.api.routes import router as inference_router
from vorta.api.speech_routes import router as speech_router

# Local imports
from vorta.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    logger.info("Starting VORTA Inference Engine...")
    
    try:
        # Initialize components using the lifespan context from dependencies
        async with lifespan_context():
            logger.info("VORTA Inference Engine started successfully")
            yield
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        logger.info("VORTA Inference Engine shutdown completed")

# Create FastAPI app
app = FastAPI(
    title="VORTA Inference Engine",
    description="Production-ready AI model inference service with caching and monitoring",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )

# Include routers
app.include_router(inference_router, prefix="/api/v1")
app.include_router(speech_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "vorta-inference-engine",
        "version": settings.version,
        "timestamp": time.time()
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "VORTA Inference Engine",
        "version": settings.version,
        "description": "Production-ready AI model inference service",
        "docs_url": "/docs" if settings.debug else "Documentation disabled in production",
        "health_url": "/health",
        "api_base": "/api/v1"
    }

# Metrics endpoint (basic)
@app.get("/metrics", tags=["Monitoring"])
async def get_basic_metrics():
    """Basic metrics endpoint"""
    return {
        "service": "vorta-inference-engine",
        "uptime": time.time(),
        "status": "running"
    }

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=1,  # Single worker for development
        log_level="info"
    )