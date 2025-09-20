#!/bin/bash
# setup-auth-service.sh - Automated Auth Service setup script

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

log_info "Setting up Auth Service for E-Commerce Microservices Platform"

# Step 1: Create directory structure
log_step "Creating Auth Service directory structure..."
mkdir -p backend/auth-service/app/{api/v1,core,models,utils,db}
log_success "Directory structure created"

# Step 2: Create Dockerfile
log_step "Creating Dockerfile..."
cat > backend/auth-service/Dockerfile << 'EOF'
# backend/auth-service/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        postgresql-client \
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
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
EOF
log_success "Dockerfile created"

# Step 3: Create requirements.txt
log_step "Creating requirements.txt..."
cat > backend/auth-service/requirements.txt << 'EOF'
# backend/auth-service/requirements.txt

# FastAPI and ASGI server
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Database
sqlalchemy==2.0.23
alembic==1.13.0
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Authentication & Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Data validation
pydantic==2.5.0
pydantic-settings==2.1.0
email-validator==2.1.0

# HTTP client for service communication
httpx==0.25.2

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

# Logging
structlog==23.2.0

# Environment management
python-dotenv==1.0.0
EOF
log_success "requirements.txt created"

# Step 4: Create __init__.py files
log_step "Creating __init__.py files..."
touch backend/auth-service/app/__init__.py
touch backend/auth-service/app/core/__init__.py
touch backend/auth-service/app/models/__init__.py
touch backend/auth-service/app/utils/__init__.py
touch backend/auth-service/app/api/__init__.py
touch backend/auth-service/app/api/v1/__init__.py
log_success "__init__.py files created"

# Step 5: Create main.py
log_step "Creating main application file..."
cat > backend/auth-service/app/main.py << 'EOF'
# backend/auth-service/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import create_tables
from app.api.v1.auth import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Auth Service...")
    await create_tables()
    print("✅ Database tables created")
    yield
    # Shutdown
    print("👋 Shutting down Auth Service...")


# Create FastAPI application
app = FastAPI(
    title="Auth Service",
    description="Authentication and user management microservice",
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
    auth_router,
    prefix="/api/v1",
    tags=["authentication"]
)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Auth Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container monitoring"""
    return JSONResponse(
        content={
            "service": "auth-service",
            "status": "healthy",
            "version": "1.0.0"
        },
        status_code=200
    )
EOF
log_success "main.py created"

# Step 6: Create config.py
log_step "Creating configuration file..."
cat > backend/auth-service/app/core/config.py << 'EOF'
# backend/auth-service/app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Auth Service"
    PROJECT_VERSION: str = "1.0.0"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/auth_db"
    )
    
    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY",
        "your-secret-key-change-in-production"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(
        os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")
    )
    
    # Security Configuration
    BCRYPT_ROUNDS: int = int(os.getenv("BCRYPT_ROUNDS", "12"))
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js frontend
        "http://localhost:8000",  # API Gateway
        "https://localhost:3000",
    ]
    
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

# Step 7: Create database.py
log_step "Creating database configuration..."
cat > backend/auth-service/app/core/database.py << 'EOF'
# backend/auth-service/app/core/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
import asyncio

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
        from app.models.user import User
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
EOF
log_success "database.py created"

# Step 8: Create user model
log_step "Creating user model..."
cat > backend/auth-service/app/models/user.py << 'EOF'
# backend/auth-service/app/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Additional profile information
    phone = Column(String(20), nullable=True)
    address = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
EOF
log_success "user.py created"

# Step 9: Create schemas
log_step "Creating Pydantic schemas..."
cat > backend/auth-service/app/models/schemas.py << 'EOF'
# backend/auth-service/app/models/schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# User Registration Schema
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8, max_length=100)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    address: Optional[str] = None


# User Login Schema
class UserLogin(BaseModel):
    email: EmailStr
    password: str


# User Response Schema (what gets returned to clients)
class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    first_name: str
    last_name: str
    phone: Optional[str] = None
    address: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


# User Update Schema
class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    address: Optional[str] = None


# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None


class RefreshToken(BaseModel):
    refresh_token: str


# Authentication Response
class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# Password Change Schema
class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


# Generic Response Schema
class MessageResponse(BaseModel):
    message: str
    success: bool = True
EOF
log_success "schemas.py created"

# Step 10: Create security utilities
log_step "Creating security utilities..."
cat > backend/auth-service/app/utils/security.py << 'EOF'
# backend/auth-service/app/utils/security.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

from app.core.config import settings


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.JWT_SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(data: dict):
    """Create a JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> dict:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Check token type
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {token_type}"
            )
        
        # Check expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        
        return payload
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def get_password_requirements() -> dict:
    """Return password requirements for frontend validation"""
    return {
        "min_length": 8,
        "max_length": 100,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_symbols": False
    }


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    requirements = get_password_requirements()
    
    if len(password) < requirements["min_length"]:
        return False, f"Password must be at least {requirements['min_length']} characters"
    
    if len(password) > requirements["max_length"]:
        return False, f"Password must be no more than {requirements['max_length']} characters"
    
    if requirements["require_uppercase"] and not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if requirements["require_lowercase"] and not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if requirements["require_numbers"] and not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    return True, "Password meets requirements"
EOF
log_success "security.py created"

# Step 11: Create API routes
log_step "Creating API routes..."
cat > backend/auth-service/app/api/v1/auth.py << 'EOF'
# backend/auth-service/app/api/v1/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.user import User
from app.models.schemas import (
    UserCreate, UserLogin, UserResponse, UserUpdate,
    AuthResponse, Token, RefreshToken, PasswordChange, MessageResponse
)
from app.utils.security import (
    hash_password, verify_password, create_access_token, create_refresh_token,
    verify_token, validate_password_strength
)


router = APIRouter()
security = HTTPBearer()


@router.post("/auth/register", response_model=AuthResponse)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    
    # Validate password strength
    is_valid, message = validate_password_strength(user_data.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = hash_password(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        hashed_password=hashed_password,
        phone=user_data.phone,
        address=user_data.address,
        is_active=True,
        is_verified=False
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    # Create tokens
    access_token = create_access_token({"sub": str(db_user.id), "email": db_user.email})
    
    return AuthResponse(
        user=UserResponse.model_validate(db_user),
        access_token=access_token,
        token_type="bearer",
        expires_in=30 * 60  # 30 minutes in seconds
    )


@router.post("/auth/login", response_model=AuthResponse)
async def login_user(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return tokens"""
    
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == login_data.email)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()
    
    # Create tokens
    access_token = create_access_token({"sub": str(user.id), "email": user.email})
    
    return AuthResponse(
        user=UserResponse.model_validate(user),
        access_token=access_token,
        token_type="bearer",
        expires_in=30 * 60  # 30 minutes in seconds
    )


@router.post("/auth/refresh", response_model=Token)
async def refresh_access_token(
    refresh_data: RefreshToken,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token"""
    
    # Verify refresh token
    payload = verify_token(refresh_data.refresh_token, token_type="refresh")
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Verify user still exists and is active
    result = await db.execute(
        select(User).where(User.id == int(user_id))
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists or is inactive"
        )
    
    # Create new access token
    access_token = create_access_token({"sub": str(user.id), "email": user.email})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=30 * 60  # 30 minutes in seconds
    )


@router.get("/auth/me", response_model=UserResponse)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get current authenticated user"""
    
    # Verify token
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == int(user_id))
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.model_validate(user)


@router.put("/auth/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Update current authenticated user"""
    
    # Verify token and get user
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")
    
    result = await db.execute(
        select(User).where(User.id == int(user_id))
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update user fields
    for field, value in user_update.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    
    user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(user)
    
    return UserResponse.model_validate(user)


@router.post("/auth/change-password", response_model=MessageResponse)
async def change_password(
    password_data: PasswordChange,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Change user password"""
    
    # Verify token and get user
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")
    
    result = await db.execute(
        select(User).where(User.id == int(user_id))
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Verify current password
    if not verify_password(password_data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password
    is_valid, message = validate_password_strength(password_data.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    # Update password
    user.hashed_password = hash_password(password_data.new_password)
    user.updated_at = datetime.utcnow()
    await db.commit()
    
    return MessageResponse(message="Password changed successfully")


@router.post("/auth/verify-token", response_model=UserResponse)
async def verify_access_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Verify token and return user (used by other services)"""
    
    # Verify token
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == int(user_id))
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return UserResponse.model_validate(user)
EOF
log_success "auth.py created"

# Step 12: Set permissions and test
log_step "Setting up permissions and testing..."
chmod +x /workspaces/microservices-app/dev.sh

log_success "Auth Service setup completed successfully! 🎉"

log_info "Next steps:"
echo "1. cd /workspaces/microservices-app/backend"
echo "2. docker-compose up --build auth-service postgres-auth -d"
echo "3. docker-compose logs -f auth-service"
echo "4. Visit http://localhost:8001/docs to see the API documentation"

log_info "Available endpoints:"
echo "- POST /api/v1/auth/register - Register new user"
echo "- POST /api/v1/auth/login - User login"
echo "- GET /api/v1/auth/me - Get current user"
echo "- PUT /api/v1/auth/me - Update user profile"
echo "- POST /api/v1/auth/change-password - Change password"
echo "- GET /health - Health check"

log_warning "Remember to update your JWT_SECRET_KEY in production!"
EOF

chmod +x setup-auth-service.sh
