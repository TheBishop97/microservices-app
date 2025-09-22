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
