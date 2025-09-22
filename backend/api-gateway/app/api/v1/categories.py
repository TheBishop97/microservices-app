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
