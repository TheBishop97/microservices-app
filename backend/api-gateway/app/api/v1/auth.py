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
