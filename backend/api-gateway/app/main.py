# backend/api-gateway/app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import httpx

from app.core.config import settings
from app.middleware.auth import AuthMiddleware
from app.middleware.logging import LoggingMiddleware
from app.api.v1.auth import router as auth_router
from app.api.v1.products import router as products_router
from app.api.v1.categories import router as categories_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting API Gateway...")
    
    # Initialize HTTP client for service communication
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    print("✅ HTTP client initialized")
    
    yield
    
    # Shutdown
    print("👋 Shutting down API Gateway...")
    await app.state.http_client.aclose()


# Create FastAPI application
app = FastAPI(
    title="API Gateway (BFF)",
    description="Backend for Frontend - Single entry point for all microservices",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)

# Include API routes
app.include_router(
    auth_router,
    prefix="/api/v1",
    tags=["authentication"]
)

app.include_router(
    products_router,
    prefix="/api/v1",
    tags=["products"]
)

app.include_router(
    categories_router,
    prefix="/api/v1",
    tags=["categories"]
)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "API Gateway (BFF)",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "services": {
            "auth_service": settings.AUTH_SERVICE_URL,
            "product_service": settings.PRODUCT_SERVICE_URL,
            "order_service": settings.ORDER_SERVICE_URL
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint that also checks downstream services"""
    services_status = {}
    
    # Check Auth Service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.AUTH_SERVICE_URL}/health")
            services_status["auth_service"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
    except Exception as e:
        services_status["auth_service"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check Product Service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.PRODUCT_SERVICE_URL}/health")
            services_status["product_service"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
    except Exception as e:
        services_status["product_service"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Determine overall health
    all_healthy = all(
        service.get("status") == "healthy" 
        for service in services_status.values()
    )
    
    return JSONResponse(
        content={
            "service": "api-gateway",
            "status": "healthy" if all_healthy else "degraded",
            "version": "1.0.0",
            "downstream_services": services_status
        },
        status_code=200 if all_healthy else 503
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url.path)
        }
    )
