.PHONY: help setup-dev start-dev stop-dev test lint format clean

help:
	@echo "🎤 VORTA AI Platform - Python Development Commands:"
	@echo ""
	@echo "  setup-dev    - Setup development environment"
	@echo "  start-dev    - Start development services (Redis, PostgreSQL, etc.)"
	@echo "  start-app    - Start VORTA application (FastAPI + Streamlit)"
	@echo "  stop-dev     - Stop development services"
	@echo "  test         - Run all tests"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format Python code"
	@echo "  clean        - Clean up temporary files"
	@echo "  install      - Install Python dependencies"

setup-dev:
	@echo "🚀 Setting up VORTA Python development environment..."
	python -m venv venv
	@echo "📦 Installing dependencies..."
	@if exist venv\\Scripts\\activate.bat ( \
		call venv\\Scripts\\activate.bat && pip install --upgrade pip && pip install -r requirements-dev.txt \
	) else ( \
		./venv/bin/activate && pip install --upgrade pip && pip install -r requirements-dev.txt \
	)
	@echo "✅ Development environment ready!"

install:
	@echo "📦 Installing Python dependencies..."
	@if exist venv\\Scripts\\activate.bat ( \
		call venv\\Scripts\\activate.bat && pip install -r requirements.txt \
	) else ( \
		source venv/bin/activate && pip install -r requirements.txt \
	)

start-dev:
	@echo "🐳 Starting VORTA development services..."
	docker-compose -f docker-compose.yml up -d
	@echo "✅ Services started!"
	@echo "📊 Services available:"
	@echo "   - Redis: localhost:6379"
	@echo "   - PostgreSQL: localhost:5432"

start-app:
	@echo "🎤 Starting VORTA AI Platform..."
	@echo "🚀 Starting FastAPI backend on port 8000..."
	@start cmd /k "cd services\\api && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
	@timeout /t 3 /nobreak >nul
	@echo "🌟 Starting Streamlit dashboard on port 8501..."
	@start cmd /k "cd frontend && streamlit run dashboard.py --server.port 8501"
	@echo "✅ VORTA Platform started!"
	@echo "📊 Access points:"
	@echo "   - Dashboard: http://localhost:8501"
	@echo "   - API: http://localhost:8000"
	@echo "   - API Docs: http://localhost:8000/docs"

stop-dev:
	@echo "🛑 Stopping development services..."
	docker-compose -f docker-compose.yml down

test:
	@echo "🧪 Running VORTA tests..."
	@if exist venv\\Scripts\\activate.bat ( \
		call venv\\Scripts\\activate.bat && python -m pytest tests/ -v --cov=services/ \
	) else ( \
		source venv/bin/activate && python -m pytest tests/ -v --cov=services/ \
	)

lint:
	@echo "🔍 Running Python code linting..."
	@if exist venv\\Scripts\\activate.bat ( \
		call venv\\Scripts\\activate.bat && python -m flake8 services/ frontend/ && python -m mypy services/ \
	) else ( \
		source venv/bin/activate && python -m flake8 services/ frontend/ && python -m mypy services/ \
	)

format:
	@echo "🎨 Formatting Python code..."
	@if exist venv\\Scripts\\activate.bat ( \
		call venv\\Scripts\\activate.bat && python -m black services/ frontend/ && python -m isort services/ frontend/ \
	) else ( \
		source venv/bin/activate && python -m black services/ frontend/ && python -m isort services/ frontend/ \
	)

clean:
	@echo "🧹 Cleaning up VORTA workspace..."
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
	@for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
	@if exist .coverage del /q .coverage
	@echo "✅ Cleanup completed!"
