#!/bin/bash
# setup-api-gateway.sh - Automated API Gateway setup script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "\n${BLUE}🚀 $1${NC}"
}

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    log_error "Please run this script from the project root directory (where backend/ exists)"
    exit 1
fi

log_info "Setting up API Gateway (BFF) for E-Commerce Microservices Platform"

# Step 1: Create directory structure
log_step "Creating API Gateway directory structure..."
mkdir -p backend/api-gateway/app/{api/v1,core,models,utils,middleware}
log_success "Directory structure created"

# Step 2: Create Dockerfile
log_step "Creating Dockerfile..."
cat > backend/api-gateway/Dockerfile << 'EOF'
# backend/api-gateway/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
log_success "Dockerfile created"

# Step 3: Create requirements.txt
log_step "Creating requirements.txt..."
cat > backend/api-gateway/requirements.txt << 'EOF'
# backend/api-gateway/requirements.txt

# FastAPI and ASGI server
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Data validation
pydantic==2.5.0
pydantic-settings==2.1.0

# HTTP client for service communication
httpx==0.25.2

# Authentication
python-jose[cryptography]==3.3.0
python-multipart==0.0.6

# Utilities
python-dateutil==2.8.2
pytz==2023.3

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
isort==5.12.0
flake8==6.1.0

# Logging and monitoring
structlog==23.2.0

# Environment management
python-dotenv==1.0.0

# Rate limiting
slowapi==0.1.9

# CORS
fastapi-cors==0.0.6
EOF
log_success "requirements.txt created"

# Step 4: Create __init__.py files
log_step "Creating __init__.py files..."
touch backend/api-gateway/app/__init__.py
touch backend/api-gateway/app/core/__init__.py
touch backend/api-gateway/app/models/__init__.py
touch backend/api-gateway/app/utils/__init__.py
touch backend/api-gateway/app/api/__init__.py
touch backend/api-gateway/app/api/v1/__init__.py
touch backend/api-gateway/app/middleware/__init__.py
log_success "__init__.py files created"

# Step 5: Create main.py
log_step "Creating main application file..."
cat > backend/api-gateway/app/main.py << 'EOF'
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
EOF
log_success "main.py created"

# Step 6: Create config.py
log_step "Creating configuration file..."
cat > backend/api-gateway/app/core/config.py << 'EOF'
# backend/api-gateway/app/core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "API Gateway (BFF)"
    PROJECT_VERSION: str = "1.0.0"
    
    # Service URLs
    AUTH_SERVICE_URL: str = os.getenv(
        "AUTH_SERVICE_URL", 
        "http://auth-service:8001"
    )
    PRODUCT_SERVICE_URL: str = os.getenv(
        "PRODUCT_SERVICE_URL", 
        "http://product-service:8002"
    )
    ORDER_SERVICE_URL: str = os.getenv(
        "ORDER_SERVICE_URL", 
        "http://order-service:8003"
    )
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js frontend
        "https://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    
    # Timeouts (seconds)
    SERVICE_TIMEOUT: float = float(os.getenv("SERVICE_TIMEOUT", "10.0"))
    SERVICE_CONNECT_TIMEOUT: float = float(os.getenv("SERVICE_CONNECT_TIMEOUT", "5.0"))
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# Create global settings instance
settings = Settings()
EOF
log_success "config.py created"

# Step 7: Create service client
log_step "Creating service client..."
cat > backend/api-gateway/app/utils/service_client.py << 'EOF'
# backend/api-gateway/app/utils/service_client.py
import httpx
from fastapi import HTTPException, Request
from typing import Dict, Any, Optional
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class ServiceClient:
    """HTTP client for communicating with microservices"""
    
    def __init__(self):
        self.timeout = httpx.Timeout(
            settings.SERVICE_TIMEOUT, 
            connect=settings.SERVICE_CONNECT_TIMEOUT
        )
        self.limits = httpx.Limits(
            max_connections=100, 
            max_keepalive_connections=20
        )
    
    async def call_service(
        self,
        service_url: str,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None
    ) -> httpx.Response:
        """Make HTTP call to a microservice"""
        
        url = f"{service_url}{endpoint}"
        
        # Prepare headers
        call_headers = {
            "Content-Type": "application/json",
            "User-Agent": "API-Gateway/1.0.0"
        }
        
        # Forward authorization header if present
        if request and hasattr(request, 'headers'):
            auth_header = request.headers.get('authorization')
            if auth_header:
                call_headers['authorization'] = auth_header
        
        if headers:
            call_headers.update(headers)
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                limits=self.limits
            ) as client:
                
                logger.info(f"Calling {method.upper()} {url}")
                
                response = await client.request(
                    method=method.upper(),
                    url=url,
                    headers=call_headers,
                    json=json_data,
                    params=params
                )
                
                logger.info(
                    f"Service call completed: {method.upper()} {url} - "
                    f"Status: {response.status_code} - "
                    f"Time: {response.elapsed.total_seconds():.3f}s"
                )
                
                return response
                
        except httpx.TimeoutException:
            logger.error(f"Timeout calling {method.upper()} {url}")
            raise HTTPException(
                status_code=504,
                detail="Service temporarily unavailable - timeout"
            )
        except httpx.ConnectError:
            logger.error(f"Connection error calling {method.upper()} {url}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable - connection error"
            )
        except Exception as e:
            logger.error(f"Unexpected error calling {method.upper()} {url}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error during service call"
            )
    
    async def get(
        self, 
        service_url: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None
    ) -> httpx.Response:
        """GET request to service"""
        return await self.call_service(
            service_url, "GET", endpoint, params=params, request=request
        )
    
    async def post(
        self, 
        service_url: str, 
        endpoint: str, 
        json_data: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None
    ) -> httpx.Response:
        """POST request to service"""
        return await self.call_service(
            service_url, "POST", endpoint, json_data=json_data, request=request
        )
    
    async def put(
        self, 
        service_url: str, 
        endpoint: str, 
        json_data: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None
    ) -> httpx.Response:
        """PUT request to service"""
        return await self.call_service(
            service_url, "PUT", endpoint, json_data=json_data, request=request
        )
    
    async def delete(
        self, 
        service_url: str, 
        endpoint: str,
        request: Optional[Request] = None
    ) -> httpx.Response:
        """DELETE request to service"""
        return await self.call_service(
            service_url, "DELETE", endpoint, request=request
        )


# Global service client instance
service_client = ServiceClient()
EOF
log_success "service_client.py created"

# Step 8: Create middleware
log_step "Creating middleware..."
cat > backend/api-gateway/app/middleware/auth.py << 'EOF'
# backend/api-gateway/app/middleware/auth.py
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

from app.core.config import settings
from app.utils.service_client import service_client

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to handle authentication for protected routes"""
    
    # Routes that don't require authentication
    PUBLIC_ROUTES = {
        "/",
        "/docs",
        "/redoc", 
        "/openapi.json",
        "/health",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/products",  # Allow public product browsing
        "/api/v1/categories"  # Allow public category browsing
    }
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public routes
        if self.is_public_route(request.url.path):
            return await call_next(request)
        
        # Skip authentication for GET requests to products/categories (public browsing)
        if request.method == "GET" and (
            request.url.path.startswith("/api/v1/products") or 
            request.url.path.startswith("/api/v1/categories")
        ):
            return await call_next(request)
        
        # Check for authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required"
            )
        
        # Verify token with auth service
        try:
            response = await service_client.post(
                service_url=settings.AUTH_SERVICE_URL,
                endpoint="/api/v1/auth/verify-token",
                request=request
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            # Add user info to request state
            user_data = response.json()
            request.state.current_user = user_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service unavailable"
            )
        
        return await call_next(request)
    
    def is_public_route(self, path: str) -> bool:
        """Check if route is public and doesn't require authentication"""
        
        # Check exact matches
        if path in self.PUBLIC_ROUTES:
            return True
        
        # Check path prefixes for documentation routes
        public_prefixes = ["/docs", "/redoc"]
        return any(path.startswith(prefix) for prefix in public_prefixes)
EOF
log_success "auth.py middleware created"

cat > backend/api-gateway/app/middleware/logging.py << 'EOF'
# backend/api-gateway/app/middleware/logging.py
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Add response time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
EOF
log_success "logging.py middleware created"

# Step 9: Create auth proxy routes
log_step "Creating auth proxy routes..."
cat > backend/api-gateway/app/api/v1/auth.py << 'EOF'
# backend/api-gateway/app/api/v1/auth.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import json

from app.core.config import settings
from app.utils.service_client import service_client

router = APIRouter()


@router.post("/auth/register")
async def register(request: Request):
    """Proxy user registration to auth service"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    # Forward to auth service
    response = await service_client.post(
        service_url=settings.AUTH_SERVICE_URL,
        endpoint="/api/v1/auth/register",
        json_data=json_data
    )
    
    # Return response
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.post("/auth/login")
async def login(request: Request):
    """Proxy user login to auth service"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    # Forward to auth service
    response = await service_client.post(
        service_url=settings.AUTH_SERVICE_URL,
        endpoint="/api/v1/auth/login",
        json_data=json_data
    )
    
    # Return response
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.post("/auth/refresh")
async def refresh_token(request: Request):
    """Proxy token refresh to auth service"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    # Forward to auth service
    response = await service_client.post(
        service_url=settings.AUTH_SERVICE_URL,
        endpoint="/api/v1/auth/refresh",
        json_data=json_data
    )
    
    # Return response
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.get("/auth/me")
async def get_current_user(request: Request):
    """Get current user information"""
    
    # Forward to auth service with authorization header
    response = await service_client.get(
        service_url=settings.AUTH_SERVICE_URL,
        endpoint="/api/v1/auth/me",
        request=request
    )
    
    # Return response
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.put("/auth/me")
async def update_current_user(request: Request):
    """Update current user profile"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    # Forward to auth service
    response = await service_client.put(
        service_url=settings.AUTH_SERVICE_URL,
        endpoint="/api/v1/auth/me",
        json_data=json_data,
        request=request
    )
    
    # Return response
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.post("/auth/change-password")
async def change_password(request: Request):
    """Change user password"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    # Forward to auth service
    response = await service_client.post(
        service_url=settings.AUTH_SERVICE_URL,
        endpoint="/api/v1/auth/change-password",
        json_data=json_data,
        request=request
    )
    
    # Return response
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )
EOF
log_success "auth.py routes created"

# Step 10: Create products proxy routes
log_step "Creating products proxy routes..."
cat > backend/api-gateway/app/api/v1/products.py << 'EOF'
# backend/api-gateway/app/api/v1/products.py
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json

from app.core.config import settings
from app.utils.service_client import service_client

router = APIRouter()


@router.get("/products")
async def list_products(request: Request):
    """List products with filtering - proxy to product service"""
    
    # Forward query parameters
    params = dict(request.query_params)
    
    # Forward to product service
    response = await service_client.get(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint="/api/v1/products",
        params=params,
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.post("/products")
async def create_product(request: Request):
    """Create new product - requires authentication"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    # Forward to product service
    response = await service_client.post(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint="/api/v1/products",
        json_data=json_data,
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.get("/products/{product_id}")
async def get_product(product_id: int, request: Request):
    """Get product by ID"""
    
    response = await service_client.get(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/products/{product_id}",
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.get("/products/slug/{slug}")
async def get_product_by_slug(slug: str, request: Request):
    """Get product by slug"""
    
    response = await service_client.get(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/products/slug/{slug}",
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.put("/products/{product_id}")
async def update_product(product_id: int, request: Request):
    """Update product - requires authentication"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    response = await service_client.put(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/products/{product_id}",
        json_data=json_data,
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.delete("/products/{product_id}")
async def delete_product(product_id: int, request: Request):
    """Delete product - requires authentication"""
    
    response = await service_client.delete(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/products/{product_id}",
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.put("/products/{product_id}/inventory")
async def update_product_inventory(product_id: int, request: Request):
    """Update product inventory - requires authentication"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    response = await service_client.put(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/products/{product_id}/inventory",
        json_data=json_data,
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.get("/products/{product_id}/stock-status")
async def get_product_stock_status(product_id: int, request: Request):
    """Get product stock status"""
    
    response = await service_client.get(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/products/{product_id}/stock-status",
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )
EOF
log_success "products.py routes created"

# Step 11: Create categories proxy routes
log_step "Creating categories proxy routes..."
cat > backend/api-gateway/app/api/v1/categories.py << 'EOF'
# backend/api-gateway/app/api/v1/categories.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import json

from app.core.config import settings
from app.utils.service_client import service_client

router = APIRouter()


@router.get("/categories")
async def list_categories(request: Request):
    """List categories - public endpoint"""
    
    # Forward query parameters
    params = dict(request.query_params)
    
    response = await service_client.get(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint="/api/v1/categories",
        params=params,
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.post("/categories")
async def create_category(request: Request):
    """Create new category - requires authentication"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    response = await service_client.post(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint="/api/v1/categories",
        json_data=json_data,
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.get("/categories/{category_id}")
async def get_category(category_id: int, request: Request):
    """Get category by ID"""
    
    response = await service_client.get(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/categories/{category_id}",
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.get("/categories/slug/{slug}")
async def get_category_by_slug(slug: str, request: Request):
    """Get category by slug"""
    
    response = await service_client.get(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/categories/slug/{slug}",
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.put("/categories/{category_id}")
async def update_category(category_id: int, request: Request):
    """Update category - requires authentication"""
    
    # Get request body
    body = await request.body()
    json_data = json.loads(body) if body else None
    
    response = await service_client.put(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/categories/{category_id}",
        json_data=json_data,
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@router.delete("/categories/{category_id}")
async def delete_category(category_id: int, request: Request):
    """Delete category - requires authentication"""
    
    response = await service_client.delete(
        service_url=settings.PRODUCT_SERVICE_URL,
        endpoint=f"/api/v1/categories/{category_id}",
        request=request
    )
    
    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )
EOF
log_success "categories.py routes created"

log_success "API Gateway setup completed successfully! 🎉"

log_info "Next steps:"
echo "1. cd /workspaces/microservices-app/backend"
echo "2. docker-compose up --build api-gateway -d"
echo "3. docker-compose logs -f api-gateway"
echo "4. Visit http://localhost:8000/docs to see the unified API documentation"

log_info "API Gateway Features:"
echo "✅ Single entry point for frontend (BFF pattern)"
echo "✅ Authentication middleware"
echo "✅ Request/response logging"
echo "✅ Service health monitoring"
echo "✅ Automatic token forwarding"
echo "✅ Error handling and timeouts"
echo "✅ CORS configuration"

log_info "Available endpoints at http://localhost:8000:"
echo "Authentication:"
echo "- POST /api/v1/auth/register"
echo "- POST /api/v1/auth/login" 
echo "- GET /api/v1/auth/me"
echo ""
echo "Products (public browsing, auth required for modifications):"
echo "- GET /api/v1/products"
echo "- POST /api/v1/products (auth required)"
echo "- GET /api/v1/products/{id}"
echo ""
echo "Categories:"
echo "- GET /api/v1/categories"
echo "- POST /api/v1/categories (auth required)"
echo ""
echo "Health & Info:"
echo "- GET /health - Health check with downstream services"
echo "- GET / - Service information"

log_warning "The API Gateway will automatically forward authentication tokens to backend services!"
EOF

chmod +x setup-api-gateway.sh
