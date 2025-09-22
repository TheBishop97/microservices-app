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
