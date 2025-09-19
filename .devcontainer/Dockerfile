# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/base:jammy

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        curl \
        wget \
        git \
        build-essential \
        ca-certificates \
        gnupg \
        lsb-release \
        vim \
        tree \
        htop

# Install Docker
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install Docker Compose standalone (for compatibility)
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Install Node.js 18 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Python 3.11 and pip
RUN apt-get -y install python3.11 python3.11-venv python3.11-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Python development tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install global Python packages for development
RUN pip install \
    black \
    flake8 \
    pytest \
    pytest-asyncio \
    httpx \
    pre-commit

# Install global Node.js packages
RUN npm install -g \
    typescript \
    @types/node \
    eslint \
    prettier \
    nodemon

# Create workspace directory
WORKDIR /workspace

# Add vscode user to docker group (for Docker-in-Docker)
RUN usermod -aG docker vscode

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
