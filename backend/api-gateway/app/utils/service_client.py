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
