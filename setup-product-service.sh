#!/bin/bash
# setup-product-service.sh - Automated Product Service setup script

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

log_info "Setting up Product Service for E-Commerce Microservices Platform"

# Step 1: Create directory structure
log_step "Creating Product Service directory structure..."
mkdir -p backend/product-service/app/{api/v1,core,models,utils,db}
log_success "Directory structure created"

# Step 2: Create Dockerfile
log_step "Creating Dockerfile..."
cat > backend/product-service/Dockerfile << 'EOF'
# backend/product-service/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        postgresql-client \
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
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
EOF
log_success "Dockerfile created"

# Step 3: Create requirements.txt
log_step "Creating requirements.txt..."
cat > backend/product-service/requirements.txt << 'EOF'
# backend/product-service/requirements.txt

# FastAPI and ASGI server
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Database
sqlalchemy==2.0.23
alembic==1.13.0
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Data validation
pydantic==2.5.0
pydantic-settings==2.1.0

# HTTP client for service communication
httpx==0.25.2

# File handling and images
python-multipart==0.0.6
pillow==10.1.0

# Utilities
python-dateutil==2.8.2
pytz==2023.3
uuid==1.30

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
isort==5.12.0
flake8==6.1.0

# Logging
structlog==23.2.0

# Environment management
python-dotenv==1.0.0

# Decimal handling for prices
decimal==1.70
EOF
log_success "requirements.txt created"

# Step 4: Create __init__.py files
log_step "Creating __init__.py files..."
touch backend/product-service/app/__init__.py
touch backend/product-service/app/core/__init__.py
touch backend/product-service/app/models/__init__.py
touch backend/product-service/app/utils/__init__.py
touch backend/product-service/app/api/__init__.py
touch backend/product-service/app/api/v1/__init__.py
log_success "__init__.py files created"

# Step 5: Create main.py
log_step "Creating main application file..."
cat > backend/product-service/app/main.py << 'EOF'
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
EOF
log_success "main.py created"

# Step 6: Create config.py
log_step "Creating configuration file..."
cat > backend/product-service/app/core/config.py << 'EOF'
# backend/product-service/app/core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Product Service"
    PROJECT_VERSION: str = "1.0.0"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/products_db"
    )
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js frontend
        "http://localhost:8000",  # API Gateway
        "https://localhost:3000",
    ]
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    UPLOAD_PATH: str = os.getenv("UPLOAD_PATH", "/app/uploads")
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # External Services (for future use)
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8001")
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# Create global settings instance
settings = Settings()
EOF
log_success "config.py created"

# Step 7: Create database.py
log_step "Creating database configuration..."
cat > backend/product-service/app/core/database.py << 'EOF'
# backend/product-service/app/core/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData

from app.core.config import settings


# Database engine
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.DEBUG,
    future=True
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Base class for database models
class Base(DeclarativeBase):
    metadata = MetaData()


# Dependency to get database session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Create all tables
async def create_tables():
    async with engine.begin() as conn:
        # Import models to register them with metadata
        from app.models.product import Product, Category, ProductImage
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
EOF
log_success "database.py created"

# Step 8: Create product models
log_step "Creating product models..."
cat > backend/product-service/app/models/product.py << 'EOF'
# backend/product-service/app/models/product.py
from sqlalchemy import Column, Integer, String, Text, Decimal, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    products = relationship("Product", back_populates="category")

    def __repr__(self):
        return f"<Category(id={self.id}, name='{self.name}')>"


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    slug = Column(String(200), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    short_description = Column(String(500), nullable=True)
    
    # Pricing
    price = Column(Decimal(10, 2), nullable=False)
    compare_price = Column(Decimal(10, 2), nullable=True)  # Original price for discounts
    cost_price = Column(Decimal(10, 2), nullable=True)     # Cost basis
    
    # Inventory
    stock_quantity = Column(Integer, default=0, nullable=False)
    track_inventory = Column(Boolean, default=True, nullable=False)
    allow_backorder = Column(Boolean, default=False, nullable=False)
    
    # Product details
    sku = Column(String(100), unique=True, nullable=True, index=True)
    barcode = Column(String(100), nullable=True)
    weight = Column(Decimal(8, 3), nullable=True)  # Weight in kg
    dimensions = Column(String(100), nullable=True)  # "L x W x H" format
    
    # SEO and metadata
    meta_title = Column(String(200), nullable=True)
    meta_description = Column(String(500), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_featured = Column(Boolean, default=False, nullable=False)
    is_digital = Column(Boolean, default=False, nullable=False)
    
    # Category relationship
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    category = relationship("Category", back_populates="products")
    images = relationship("ProductImage", back_populates="product", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', price={self.price})>"


class ProductImage(Base):
    __tablename__ = "product_images"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    url = Column(String(500), nullable=False)
    alt_text = Column(String(200), nullable=True)
    is_primary = Column(Boolean, default=False, nullable=False)
    sort_order = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    product = relationship("Product", back_populates="images")

    def __repr__(self):
        return f"<ProductImage(id={self.id}, product_id={self.product_id})>"
EOF
log_success "product.py created"

# Step 9: Create schemas
log_step "Creating Pydantic schemas..."
cat > backend/product-service/app/models/schemas.py << 'EOF'
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
EOF
log_success "schemas.py created"

# Step 10: Create utility functions
log_step "Creating utility functions..."
cat > backend/product-service/app/utils/helpers.py << 'EOF'
# backend/product-service/app/utils/helpers.py
import re
from typing import Optional
from decimal import Decimal


def create_slug(text: str) -> str:
    """Create a URL-friendly slug from text"""
    # Convert to lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def format_price(price: Decimal) -> str:
    """Format price for display"""
    return f"${price:.2f}"


def calculate_discount_percentage(original_price: Decimal, sale_price: Decimal) -> Optional[float]:
    """Calculate discount percentage"""
    if original_price <= 0 or sale_price >= original_price:
        return None
    
    discount = original_price - sale_price
    percentage = (discount / original_price) * 100
    return round(percentage, 1)


def is_in_stock(product) -> bool:
    """Check if product is in stock"""
    if not product.track_inventory:
        return True
    
    if product.allow_backorder:
        return True
    
    return product.stock_quantity > 0


def validate_sku(sku: str) -> bool:
    """Validate SKU format"""
    if not sku:
        return False
    
    # SKU should be alphanumeric with optional hyphens/underscores
    pattern = r'^[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, sku)) and 3 <= len(sku) <= 100


def parse_dimensions(dimensions: str) -> Optional[dict]:
    """Parse dimensions string into length, width, height"""
    if not dimensions:
        return None
    
    # Expected format: "L x W x H" (e.g., "10.5 x 8.2 x 3.0")
    pattern = r'^(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)$'
    match = re.match(pattern, dimensions.strip(), re.IGNORECASE)
    
    if match:
        return {
            "length": float(match.group(1)),
            "width": float(match.group(2)),
            "height": float(match.group(3))
        }
    
    return None


def generate_sku_suggestion(name: str, category: str = None) -> str:
    """Generate a SKU suggestion based on product name and category"""
    # Take first 3 letters of category (if provided)
    category_part = ""
    if category:
        category_part = re.sub(r'[^\w]', '', category)[:3].upper()
    
    # Take first letters of each word in product name
    name_words = re.findall(r'\w+', name.upper())
    name_part = ''.join(word[0] for word in name_words[:3])
    
    # Add a number for uniqueness (should be checked against existing SKUs)
    base_sku = f"{category_part}{name_part}"
    return base_sku if base_sku else "PROD"


class PriceCalculator:
    """Helper class for price calculations"""
    
    @staticmethod
    def add_tax(price: Decimal, tax_rate: Decimal) -> Decimal:
        """Add tax to price"""
        return price * (1 + tax_rate)
    
    @staticmethod
    def apply_discount(price: Decimal, discount_percent: float) -> Decimal:
        """Apply percentage discount to price"""
        if not (0 <= discount_percent <= 100):
            raise ValueError("Discount percentage must be between 0 and 100")
        
        discount_multiplier = Decimal(str(1 - (discount_percent / 100)))
        return price * discount_multiplier
    
    @staticmethod
    def calculate_margin(cost: Decimal, price: Decimal) -> float:
        """Calculate profit margin percentage"""
        if cost <= 0 or price <= 0:
            return 0.0
        
        margin = ((price - cost) / price) * 100
        return round(float(margin), 2)
EOF
log_success "helpers.py created"

# Step 11: Create product API routes
log_step "Creating product API routes..."
cat > backend/product-service/app/api/v1/products.py << 'EOF'
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
EOF
log_success "products.py created"

# Step 12: Create categories API routes
log_step "Creating categories API routes..."
cat > backend/product-service/app/api/v1/categories.py << 'EOF'
# backend/product-service/app/api/v1/categories.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional

from app.core.database import get_db
from app.models.product import Category
from app.models.schemas import (
    CategoryCreate, CategoryUpdate, CategoryResponse, CategoryListResponse,
    PaginationInfo, MessageResponse
)


router = APIRouter()


@router.get("/categories", response_model=CategoryListResponse)
async def list_categories(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    is_active: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """List categories with pagination"""
    
    # Build query
    query = select(Category)
    
    # Apply filters
    if is_active is not None:
        query = query.where(Category.is_active == is_active)
    
    # Get total count
    count_query = select(func.count()).select_from(Category)
    if is_active is not None:
        count_query = count_query.where(Category.is_active == is_active)
    
    result = await db.execute(count_query)
    total_items = result.scalar()
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    # Order by name
    query = query.order_by(Category.name)
    
    # Execute query
    result = await db.execute(query)
    categories = result.scalars().all()
    
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
    
    return CategoryListResponse(
        categories=[CategoryResponse.model_validate(c) for c in categories],
        pagination=pagination
    )


@router.post("/categories", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(
    category_data: CategoryCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new category"""
    
    # Check if name already exists
    result = await db.execute(
        select(Category).where(Category.name == category_data.name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category with this name already exists"
        )
    
    # Check if slug already exists
    result = await db.execute(
        select(Category).where(Category.slug == category_data.slug)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category with this slug already exists"
        )
    
    # Create category
    db_category = Category(**category_data.model_dump())
    
    db.add(db_category)
    await db.commit()
    await db.refresh(db_category)
    
    return CategoryResponse.model_validate(db_category)


@router.get("/categories/{category_id}", response_model=CategoryResponse)
async def get_category(
    category_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific category by ID"""
    
    result = await db.execute(
        select(Category).where(Category.id == category_id)
    )
    category = result.scalar_one_or_none()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    return CategoryResponse.model_validate(category)


@router.get("/categories/slug/{slug}", response_model=CategoryResponse)
async def get_category_by_slug(
    slug: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific category by slug"""
    
    result = await db.execute(
        select(Category).where(Category.slug == slug)
    )
    category = result.scalar_one_or_none()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    return CategoryResponse.model_validate(category)


@router.put("/categories/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: int,
    category_update: CategoryUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a category"""
    
    # Get existing category
    result = await db.execute(
        select(Category).where(Category.id == category_id)
    )
    category = result.scalar_one_or_none()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    # Update fields
    update_data = category_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(category, field, value)
    
    await db.commit()
    await db.refresh(category)
    
    return CategoryResponse.model_validate(category)


@router.delete("/categories/{category_id}", response_model=MessageResponse)
async def delete_category(
    category_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a category"""
    
    result = await db.execute(
        select(Category).where(Category.id == category_id)
    )
    category = result.scalar_one_or_none()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    await db.delete(category)
    await db.commit()
    
    return MessageResponse(message="Category deleted successfully")
EOF
log_success "categories.py created"

log_success "Product Service setup completed successfully! 🎉"

log_info "Next steps:"
echo "1. cd /workspaces/microservices-app/backend"
echo "2. docker-compose up --build product-service postgres-products -d"
echo "3. docker-compose logs -f product-service"
echo "4. Visit http://localhost:8002/docs to see the API documentation"

log_info "Available endpoints:"
echo "Products:"
echo "- GET /api/v1/products - List products with filtering"
echo "- POST /api/v1/products - Create new product"
echo "- GET /api/v1/products/{id} - Get product by ID"
echo "- PUT /api/v1/products/{id} - Update product"
echo "- DELETE /api/v1/products/{id} - Delete product"
echo "- PUT /api/v1/products/{id}/inventory - Update inventory"
echo ""
echo "Categories:"
echo "- GET /api/v1/categories - List categories"
echo "- POST /api/v1/categories - Create new category"
echo "- GET /api/v1/categories/{id} - Get category by ID"
echo "- PUT /api/v1/categories/{id} - Update category"
echo "- DELETE /api/v1/categories/{id} - Delete category"
EOF

chmod +x setup-product-service.sh
