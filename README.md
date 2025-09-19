# Decoupled Microservices E-Commerce Platform

A modern, scalable e-commerce platform demonstrating industry-standard practices in full-stack development, microservices architecture, and cloud deployment.

## Architecture Overview

### System Design
- **Frontend**: Next.js (SSR) hosted on Netlify CDN
- **Backend**: FastAPI microservices on IBM LinuxONE VPS
- **Communication**: Decoupled HTTP API through Backend for Frontend pattern
- **Containerization**: Docker with Docker Compose orchestration
- **Reverse Proxy**: Nginx with SSL termination

### Services Architecture
```
Frontend (Netlify) → API Gateway (BFF) → Internal Services
                           ↓
                    ┌─────────────────┐
                    │   Auth Service  │ (JWT, Users)
                    │ Product Service │ (Catalog, Inventory)  
                    │  Order Service  │ (Cart, Orders)
                    └─────────────────┘
```

## Quick Start

### Prerequisites
- GitHub account with Codespaces access
- Git installed locally (if cloning)

### Development Setup

1. **Open in GitHub Codespaces**:
   ```bash
   # From GitHub repository page, click "Code" → "Codespaces" → "Create codespace"
   # Or use GitHub CLI:
   gh codespace create -r your-username/microservices-app
   ```

2. **Verify Environment**:
   ```bash
   # Check installations
   python --version    # Should be 3.11+
   node --version      # Should be 18+
   docker --version    # Should be 20+
   docker-compose --version
   ```

3. **Start Backend Services**:
   ```bash
   cd backend
   docker-compose up --build
   ```

4. **Start Frontend Development**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

5. **Access Services**:
   - Frontend: http://localhost:3000
   - API Gateway: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Project Structure

```
microservices-app/
├── .devcontainer/         # Codespaces configuration
├── .github/workflows/     # CI/CD pipelines
├── frontend/              # Next.js application
│   ├── src/
│   ├── public/
│   └── netlify.toml
└── backend/               # Microservices
    ├── api-gateway/       # BFF + Service Router
    ├── auth-service/      # Identity Management
    ├── product-service/   # Product Catalog
    ├── order-service/     # Cart & Orders
    └── docker-compose.yml
```

## Development Workflow

### Backend Development
```bash
# Start all services in development mode
cd backend
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Rebuild specific service
docker-compose up --build auth-service

# View logs
docker-compose logs -f api-gateway

# Run tests
docker-compose exec auth-service python -m pytest
```

### Frontend Development
```bash
cd frontend
npm run dev        # Development server
npm run build      # Production build
npm run lint       # Code linting
npm run type-check # TypeScript checking
```

## API Documentation

### Authentication Endpoints
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/refresh` - Token refresh

### Product Endpoints
- `GET /api/v1/products` - List products
- `GET /api/v1/products/{id}` - Get product details
- `POST /api/v1/products` - Create product (admin)

### Order Endpoints
- `GET /api/v1/cart` - Get user cart
- `POST /api/v1/cart/items` - Add item to cart
- `POST /api/v1/orders` - Create order from cart

## Deployment

### Backend (IBM LinuxONE VPS)
Automatic deployment via GitHub Actions on push to `main` branch.

### Frontend (Netlify)
Automatic deployment via Netlify integration on push to `main` branch.

## Environment Variables

### Backend (.env)
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/ecommerce

# JWT
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Gateway
API_GATEWAY_HOST=0.0.0.0
API_GATEWAY_PORT=8000
```

### Frontend (.env.local)
```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://your-vps-domain.com
NEXT_PUBLIC_ENVIRONMENT=development
```

## Contributing

1. Create feature branch from `main`
2. Make changes in appropriate service
3. Test locally with Docker Compose
4. Submit pull request

## Learning Objectives

- Microservices architecture patterns
- Container orchestration with Docker
- API design and documentation
- Authentication and authorization
- Frontend-backend decoupling
- Cloud deployment strategies
- CI/CD pipeline automation

## Technology Stack

**Frontend**:
- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind CSS
- Netlify

**Backend**:
- FastAPI (Python)
- PostgreSQL
- Docker & Docker Compose
- Nginx
- JWT Authentication
- IBM LinuxONE (Ubuntu)

**DevOps**:
- GitHub Codespaces
- GitHub Actions
- Docker Hub
- SSL/TLS encryption
