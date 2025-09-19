#!/bin/bash
# dev.sh - Development helper script for microservices e-commerce platform

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

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

show_help() {
    echo "Development Helper Script for Microservices E-Commerce Platform"
    echo
    echo "Usage: ./dev.sh [COMMAND]"
    echo
    echo "Commands:"
    echo "  setup           - Initial project setup"
    echo "  start           - Start all services (backend + frontend)"
    echo "  backend         - Start only backend services"
    echo "  frontend        - Start only frontend"
    echo "  build           - Build all services"
    echo "  test            - Run all tests"
    echo "  clean           - Clean up containers and volumes"
    echo "  logs [service]  - View logs (optional: specific service)"
    echo "  shell [service] - Open shell in service container"
    echo "  check           - Check system requirements"
    echo "  help            - Show this help message"
    echo
    echo "Examples:"
    echo "  ./dev.sh start"
    echo "  ./dev.sh logs api-gateway"
    echo "  ./dev.sh shell auth-service"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        log_success "Python $PYTHON_VERSION installed"
    else
        log_error "Python not found"
    fi
    
    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log_success "Node.js $NODE_VERSION installed"
    else
        log_error "Node.js not found"
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        log_success "Docker $DOCKER_VERSION installed"
    else
        log_error "Docker not found"
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
        log_success "Docker Compose $COMPOSE_VERSION installed"
    else
        log_error "Docker Compose not found"
    fi
}

setup_project() {
    log_info "Setting up project..."
    
    # Create necessary directories
    mkdir -p backend/{api-gateway,auth-service,product-service,order-service}/app/{api,core,models}
    mkdir -p frontend/src/{components,pages,styles,lib}
    mkdir -p frontend/public
    
    # Create placeholder files to maintain directory structure
    touch backend/docker-compose.yml
    touch backend/docker-compose.override.yml
    touch backend/.env.example
    touch frontend/package.json
    touch frontend/netlify.toml
    
    log_success "Project structure created"
    
    # Install frontend dependencies if package.json exists and has content
    if [ -s "frontend/package.json" ]; then
        log_info "Installing frontend dependencies..."
        cd frontend && npm install && cd ..
        log_success "Frontend dependencies installed"
    fi
}

start_backend() {
    log_info "Starting backend services..."
    cd backend
    if [ -f "docker-compose.yml" ] && [ -s "docker-compose.yml" ]; then
        docker-compose up --build -d
        log_success "Backend services started"
        log_info "API Gateway: http://localhost:8000"
        log_info "API Docs: http://localhost:8000/docs"
    else
        log_warning "docker-compose.yml not found or empty. Please set up backend services first."
    fi
    cd ..
}

start_frontend() {
    log_info "Starting frontend development server..."
    cd frontend
    if [ -f "package.json" ] && [ -s "package.json" ]; then
        if [ ! -d "node_modules" ]; then
            log_info "Installing dependencies first..."
            npm install
        fi
        npm run dev &
        log_success "Frontend server started at http://localhost:3000"
    else
        log_warning "package.json not found or empty. Please set up frontend first."
    fi
    cd ..
}

build_services() {
    log_info "Building all services..."
    
    # Build backend
    cd backend
    if [ -f "docker-compose.yml" ] && [ -s "docker-compose.yml" ]; then
        docker-compose build
        log_success "Backend services built"
    fi
    cd ..
    
    # Build frontend
    cd frontend
    if [ -f "package.json" ] && [ -s "package.json" ]; then
        npm run build
        log_success "Frontend built"
    fi
    cd ..
}

run_tests() {
    log_info "Running tests..."
    
    # Backend tests
    cd backend
    if [ -f "docker-compose.yml" ] && [ -s "docker-compose.yml" ]; then
        docker-compose exec -T auth-service python -m pytest || log_warning "Auth service tests failed"
        docker-compose exec -T product-service python -m pytest || log_warning "Product service tests failed"
        docker-compose exec -T order-service python -m pytest || log_warning "Order service tests failed"
    fi
    cd ..
    
    # Frontend tests
    cd frontend
    if [ -f "package.json" ] && [ -s "package.json" ]; then
        npm test || log_warning "Frontend tests failed"
    fi
    cd ..
}

clean_containers() {
    log_info "Cleaning up containers and volumes..."
    cd backend
    docker-compose down -v --remove-orphans
    docker system prune -f
    cd ..
    log_success "Cleanup completed"
}

view_logs() {
    local service=$1
    cd backend
    if [ -z "$service" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$service"
    fi
    cd ..
}

open_shell() {
    local service=$1
    if [ -z "$service" ]; then
        log_error "Please specify a service name"
        log_info "Available services: api-gateway, auth-service, product-service, order-service"
        exit 1
    fi
    
    cd backend
    docker-compose exec "$service" /bin/bash
    cd ..
}

# Main command handling
case "$1" in
    setup)
        setup_project
        ;;
    start)
        start_backend
        sleep 5
        start_frontend
        ;;
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    build)
        build_services
        ;;
    test)
        run_tests
        ;;
    clean)
        clean_containers
        ;;
    logs)
        view_logs "$2"
        ;;
    shell)
        open_shell "$2"
        ;;
    check)
        check_requirements
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
