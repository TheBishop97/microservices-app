# backend/product-service/app/api/v1/products.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import selectinload
from typing import Optional
from decimal import Decimal

from app.core.database import get_db
from app.models.product import Product, ProductImage
from app.models.schemas import (
    ProductCreate, ProductUpdate, ProductResponse, ProductListResponse,
    PaginationInfo, ProductFilters, InventoryUpdate, MessageResponse
)
from app.utils.helpers import create_slug, is_in_stock


router = APIRouter()


@router.get("/products", response_model=ProductListResponse)
async def list_products(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category_id: Optional[int] = Query(None),
    min_price: Optional[Decimal] = Query(None, ge=0),
    max_price: Optional[Decimal] = Query(None, ge=0),
    is_featured: Optional[bool] = Query(None),
    is_active: Optional[bool] = Query(True),
    in_stock: Optional[bool] = Query(None),
    search: Optional[str] = Query(None, max_length=200),
    db: AsyncSession = Depends(get_db)
):
    """List products with filtering and pagination"""
    
    # Build query
    query = select(Product).options(
        selectinload(Product.category),
        selectinload(Product.images)
    )
    
    # Apply filters
    filters = []
    
    if category_id is not None:
        filters.append(Product.category_id == category_id)
    
    if min_price is not None:
        filters.append(Product.price >= min_price)
    
    if max_price is not None:
        filters.append(Product.price <= max_price)
    
    if is_featured is not None:
        filters.append(Product.is_featured == is_featured)
    
    if is_active is not None:
        filters.append(Product.is_active == is_active)
    
    if in_stock is not None:
        if in_stock:
            filters.append(
                or_(
                    Product.track_inventory == False,
                    Product.allow_backorder == True,
                    Product.stock_quantity > 0
                )
            )
        else:
            filters.append(
                and_(
                    Product.track_inventory == True,
                    Product.allow_backorder == False,
                    Product.stock_quantity <= 0
                )
            )
    
    if search:
        search_term = f"%{search}%"
        filters.append(
            or_(
                Product.name.ilike(search_term),
                Product.description.ilike(search_term),
                Product.sku.ilike(search_term)
            )
        )
    
    if filters:
        query = query.where(and_(*filters))
    
    # Get total count
    count_query = select(func.count()).select_from(Product)
    if filters:
        count_query = count_query.where(and_(*filters))
    
    result = await db.execute(count_query)
    total_items = result.scalar()
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    # Order by created_at descending
    query = query.order_by(Product.created_at.desc())
    
    # Execute query
    result = await db.execute(query)
    products = result.scalars().all()
    
    # Calculate pagination info
    total_pages = (total_items + page_size - 1) // page_size
    
    pagination = PaginationInfo(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1
    )
    
    return ProductListResponse(
        products=[ProductResponse.model_validate(p) for p in products],
        pagination=pagination
    )


@router.post("/products", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    product_data: ProductCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new product"""
    
    # Check if slug already exists
    result = await db.execute(
        select(Product).where(Product.slug == product_data.slug)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Product with this slug already exists"
        )
    
    # Check if SKU already exists (if provided)
    if product_data.sku:
        result = await db.execute(
            select(Product).where(Product.sku == product_data.sku)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Product with this SKU already exists"
            )
    
    # Create product
    product_dict = product_data.model_dump(exclude={"images"})
    db_product = Product(**product_dict)
    
    db.add(db_product)
    await db.commit()
    await db.refresh(db_product)
    
    # Add images if provided
    for img_data in product_data.images:
        db_image = ProductImage(
            product_id=db_product.id,
            **img_data.model_dump()
        )
        db.add(db_image)
    
    await db.commit()
    
    # Fetch product with relationships
    result = await db.execute(
        select(Product).options(
            selectinload(Product.category),
            selectinload(Product.images)
        ).where(Product.id == db_product.id)
    )
    product = result.scalar_one()
    
    return ProductResponse.model_validate(product)


@router.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific product by ID"""
    
    result = await db.execute(
        select(Product).options(
            selectinload(Product.category),
            selectinload(Product.images)
        ).where(Product.id == product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    return ProductResponse.model_validate(product)


@router.get("/products/slug/{slug}", response_model=ProductResponse)
async def get_product_by_slug(
    slug: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific product by slug"""
    
    result = await db.execute(
        select(Product).options(
            selectinload(Product.category),
            selectinload(Product.images)
        ).where(Product.slug == slug)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    return ProductResponse.model_validate(product)


@router.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product_update: ProductUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a product"""
    
    # Get existing product
    result = await db.execute(
        select(Product).where(Product.id == product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    # Update fields
    update_data = product_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(product, field, value)
    
    await db.commit()
    await db.refresh(product)
    
    # Fetch with relationships
    result = await db.execute(
        select(Product).options(
            selectinload(Product.category),
            selectinload(Product.images)
        ).where(Product.id == product_id)
    )
    updated_product = result.scalar_one()
    
    return ProductResponse.model_validate(updated_product)


@router.delete("/products/{product_id}", response_model=MessageResponse)
async def delete_product(
    product_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a product"""
    
    result = await db.execute(
        select(Product).where(Product.id == product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    await db.delete(product)
    await db.commit()
    
    return MessageResponse(message="Product deleted successfully")


@router.put("/products/{product_id}/inventory", response_model=ProductResponse)
async def update_inventory(
    product_id: int,
    inventory_update: InventoryUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update product inventory"""
    
    result = await db.execute(
        select(Product).where(Product.id == product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    # Update inventory fields
    product.stock_quantity = inventory_update.stock_quantity
    if inventory_update.track_inventory is not None:
        product.track_inventory = inventory_update.track_inventory
    if inventory_update.allow_backorder is not None:
        product.allow_backorder = inventory_update.allow_backorder
    
    await db.commit()
    
    # Fetch with relationships
    result = await db.execute(
        select(Product).options(
            selectinload(Product.category),
            selectinload(Product.images)
        ).where(Product.id == product_id)
    )
    updated_product = result.scalar_one()
    
    return ProductResponse.model_validate(updated_product)


@router.get("/products/{product_id}/stock-status")
async def get_stock_status(
    product_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get product stock status"""
    
    result = await db.execute(
        select(Product).where(Product.id == product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    in_stock = is_in_stock(product)
    
    return {
        "product_id": product_id,
        "in_stock": in_stock,
        "stock_quantity": product.stock_quantity,
        "track_inventory": product.track_inventory,
        "allow_backorder": product.allow_backorder
    }
