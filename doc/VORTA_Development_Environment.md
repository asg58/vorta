# VORTA Development Environment Setup

## 🐳 Docker & Container Setup

### **1. Docker Desktop Installatie**

**Windows:**

```powershell
# Download Docker Desktop voor Windows
# https://docs.docker.com/desktop/install/windows-install/

# Verificeer installatie
docker --version
docker-compose --version
```

**macOS:**

```bash
# Via Homebrew
brew install --cask docker

# Of download van: https://docs.docker.com/desktop/install/mac-install/

# Verificeer installatie
docker --version
docker-compose --version
```

**Linux (Ubuntu/Debian):**

```bash
# Update package index
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verificeer installatie
docker --version
docker-compose --version
```

### **2. Python Development Environment**

```bash
# Python 3.11+ installeren
# Windows: Download van python.org
# macOS: brew install python@3.11
# Linux: sudo apt-get install python3.11 python3.11-venv

# Verificeer Python versie
python --version  # Should be 3.11+

# Maak virtual environment voor VORTA development
cd vorta-platform
python -m venv venv

# Activeer virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install development dependencies
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    torch==2.1.0 \
    transformers==4.35.0 \
    faiss-cpu==1.7.4 \
    redis==5.0.1 \
    prometheus-client==0.19.0 \
    httpx==0.25.2 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    black==23.11.0 \
    isort==5.12.0 \
    mypy==1.7.1 \
    pre-commit==3.5.0

# Save dependencies
pip freeze > requirements-dev.txt
```

### **3. VS Code Setup & Extensions**

**VS Code Installatie:**

```bash
# Download van: https://code.visualstudio.com/

# Of via package manager:
# Windows: winget install Microsoft.VisualStudioCode
# macOS: brew install --cask visual-studio-code
# Linux: sudo snap install code --classic
```

**Essential Extensions:**

```bash
# Install via VS Code command palette (Ctrl+Shift+P)
# Of via command line:

code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension ms-python.mypy-type-checker
code --install-extension ms-vscode.vscode-typescript-next
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
code --install-extension ms-vscode.docker
code --install-extension hashicorp.terraform
code --install-extension redhat.vscode-yaml
code --install-extension ms-vscode.test-adapter-converter
code --install-extension github.copilot
code --install-extension ms-vscode.github-copilot-chat
```

**VS Code Workspace Settings:**

```json
# .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true,
        "**/.git": false
    },
    "docker.defaultPlatform": "linux",
    "yaml.schemas": {
        "https://json.schemastore.org/github-workflow.json": ".github/workflows/*.yml",
        "https://json.schemastore.org/kustomization.json": "kustomization.yaml"
    }
}
```

### **4. Kubernetes Development Tools**

```bash
# kubectl installatie
# Windows (via Chocolatey):
choco install kubernetes-cli

# macOS (via Homebrew):
brew install kubectl

# Linux:
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Helm installatie
# Windows:
choco install kubernetes-helm

# macOS:
brew install helm

# Linux:
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# minikube voor local Kubernetes
# Windows:
choco install minikube

# macOS:
brew install minikube

# Linux:
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Verificeer installaties
kubectl version --client
helm version
minikube version
```

### **5. Local Development Stack**

**docker-compose.yml voor development:**

```yaml
# infrastructure/docker/docker-compose.dev.yml
version: "3.8"

services:
  # Redis voor caching
  redis:
    image: redis:7-alpine
    container_name: vorta-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL voor metadata
  postgres:
    image: postgres:15-alpine
    container_name: vorta-postgres
    environment:
      POSTGRES_DB: vorta
      POSTGRES_USER: vorta
      POSTGRES_PASSWORD: vorta_dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U vorta"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Prometheus voor metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: vorta-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"

  # Grafana voor dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: vorta-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

  # Jaeger voor tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: vorta-jaeger
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      COLLECTOR_OTLP_ENABLED: true

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

**Prometheus configuratie:**

```yaml
# infrastructure/docker/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "vorta-inference-engine"
    static_configs:
      - targets: ["host.docker.internal:8000"]
    metrics_path: "/metrics"
    scrape_interval: 5s

  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
```

### **6. Development Workflow Setup**

**Pre-commit hooks:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
```

**Makefile voor development commands:**

```makefile
# Makefile
.PHONY: help setup-dev start-dev stop-dev test lint format clean

help:
	@echo "VORTA Development Commands:"
	@echo "  setup-dev    - Setup development environment"
	@echo "  start-dev    - Start development services"
	@echo "  stop-dev     - Stop development services"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean up temporary files"

setup-dev:
	@echo "🚀 Setting up VORTA development environment..."
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements-dev.txt
	pre-commit install
	@echo "✅ Development environment ready!"

start-dev:
	@echo "🐳 Starting development services..."
	docker-compose -f infrastructure/docker/docker-compose.dev.yml up -d
	@echo "✅ Services started!"
	@echo "📊 Grafana: http://localhost:3000 (admin/admin)"
	@echo "📈 Prometheus: http://localhost:9090"
	@echo "🔍 Jaeger: http://localhost:16686"

stop-dev:
	@echo "🛑 Stopping development services..."
	docker-compose -f infrastructure/docker/docker-compose.dev.yml down

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=services/

lint:
	@echo "🔍 Running linting..."
	mypy services/
	black --check services/
	isort --check-only services/

format:
	@echo "🎨 Formatting code..."
	black services/
	isort services/

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
```

## 🧪 Environment Verification

**Test je development environment:**

```bash
# 1. Clone repository (als nog niet gedaan)
git clone https://github.com/your-username/vorta-platform.git
cd vorta-platform

# 2. Setup development environment
make setup-dev

# 3. Start development services
make start-dev

# 4. Verify services
curl http://localhost:3000  # Grafana
curl http://localhost:9090  # Prometheus
curl http://localhost:16686 # Jaeger

# 5. Check Docker containers
docker ps

# 6. Test Python environment
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -c "import fastapi, torch, transformers; print('✅ All imports successful')"
```

## ✅ Development Environment Checklist

Na deze setup zou je moeten hebben:

- [ ] ✅ **Docker Desktop** running
- [ ] ✅ **Python 3.11+** with virtual environment
- [ ] ✅ **VS Code** with essential extensions
- [ ] ✅ **kubectl, helm, minikube** installed
- [ ] ✅ **Development services** running (Redis, PostgreSQL, Prometheus, Grafana)
- [ ] ✅ **Pre-commit hooks** configured
- [ ] ✅ **Makefile commands** working

**Ready voor Stap 3: Inference Engine MVP Implementation! 🚀**
