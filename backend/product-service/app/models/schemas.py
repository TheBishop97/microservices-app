# backend/product-service/app/models/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
from decimal import Decimal


# Category Schemas
class CategoryBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    slug: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    image_url: Optional[str] = None
    is_active: bool = True


class CategoryCreate(CategoryBase):
    pass


class CategoryUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    image_url: Optional[str] = None
    is_active: Optional[bool] = None


class CategoryResponse(CategoryBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Product Image Schemas
class ProductImageBase(BaseModel):
    url: str
    alt_text: Optional[str] = None
    is_primary: bool = False
    sort_order: int = 0


class ProductImageCreate(ProductImageBase):
    pass


class ProductImageResponse(ProductImageBase):
    id: int
    product_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Product Schemas
class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    slug: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    short_description: Optional[str] = Field(None, max_length=500)
    price: Decimal = Field(..., gt=0)
    compare_price: Optional[Decimal] = Field(None, gt=0)
    cost_price: Optional[Decimal] = Field(None, ge=0)
    stock_quantity: int = Field(0, ge=0)
    track_inventory: bool = True
    allow_backorder: bool = False
    sku: Optional[str] = Field(None, max_length=100)
    barcode: Optional[str] = Field(None, max_length=100)
    weight: Optional[Decimal] = Field(None, gt=0)
    dimensions: Optional[str] = Field(None, max_length=100)
    meta_title: Optional[str] = Field(None, max_length=200)
    meta_description: Optional[str] = Field(None, max_length=500)
    is_active: bool = True
    is_featured: bool = False
    is_digital: bool = False
    category_id: Optional[int] = None

    @validator('compare_price')
    def compare_price_must_be_greater_than_price(cls, v, values):
        if v is not None and 'price' in values and v <= values['price']:
            raise ValueError('compare_price must be greater than price')
        return v


class ProductCreate(ProductBase):
    images: List[ProductImageCreate] = []


class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    short_description: Optional[str] = Field(None, max_length=500)
    price: Optional[Decimal] = Field(None, gt=0)
    compare_price: Optional[Decimal] = Field(None, gt=0)
    cost_price: Optional[Decimal] = Field(None, ge=0)
    stock_quantity: Optional[int] = Field(None, ge=0)
    track_inventory: Optional[bool] = None
    allow_backorder: Optional[bool] = None
    sku: Optional[str] = Field(None, max_length=100)
    barcode: Optional[str] = Field(None, max_length=100)
    weight: Optional[Decimal] = Field(None, gt=0)
    dimensions: Optional[str] = Field(None, max_length=100)
    meta_title: Optional[str] = Field(None, max_length=200)
    meta_description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    is_featured: Optional[bool] = None
    is_digital: Optional[bool] = None
    category_id: Optional[int] = None


class ProductResponse(ProductBase):
    id: int
    category: Optional[CategoryResponse] = None
    images: List[ProductImageResponse] = []
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Pagination and List Responses
class PaginationInfo(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool


class ProductListResponse(BaseModel):
    products: List[ProductResponse]
    pagination: PaginationInfo


class CategoryListResponse(BaseModel):
    categories: List[CategoryResponse]
    pagination: PaginationInfo


# Search and Filter Schemas
class ProductFilters(BaseModel):
    category_id: Optional[int] = None
    min_price: Optional[Decimal] = None
    max_price: Optional[Decimal] = None
    is_featured: Optional[bool] = None
    is_active: Optional[bool] = None
    in_stock: Optional[bool] = None
    search: Optional[str] = None


# Inventory Update Schema
class InventoryUpdate(BaseModel):
    stock_quantity: int = Field(..., ge=0)
    track_inventory: Optional[bool] = None
    allow_backorder: Optional[bool] = None


# Generic Response Schema
class MessageResponse(BaseModel):
    message: str
    success: bool = True
