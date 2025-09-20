# backend/product-service/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import create_tables
from app.api.v1.products import router as products_router
from app.api.v1.categories import router as categories_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Product Service...")
    await create_tables()
    print("✅ Database tables created")
    yield
    # Shutdown
    print("👋 Shutting down Product Service...")


# Create FastAPI application
app = FastAPI(
    title="Product Service",
    description="Product catalog and inventory management microservice",
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

# Include API routes
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
        "service": "Product Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container monitoring"""
    return JSONResponse(
        content={
            "service": "product-service",
            "status": "healthy",
            "version": "1.0.0"
        },
        status_code=200
    )
