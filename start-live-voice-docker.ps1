#!/usr/bin/env pwsh

# VORTA Live Voice Conversation - Docker Startup Script
# Starts complete live voice conversation system with Docker

Write-Host "🎙️ VORTA Live Voice Conversation - Docker Startup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if Docker is running
$dockerRunning = docker info 2>$null
if (-not $dockerRunning) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker is running" -ForegroundColor Green

# Stop any existing VORTA live voice containers
Write-Host "`n🛑 Stopping existing VORTA live voice containers..." -ForegroundColor Yellow
docker-compose -f docker-compose-live-voice.yml down 2>$null

# Clean up any orphaned containers
Write-Host "🧹 Cleaning up orphaned containers..." -ForegroundColor Yellow
docker container prune -f 2>$null

# Build and start the live voice services
Write-Host "`n🏗️ Building and starting VORTA Live Voice services..." -ForegroundColor Cyan
docker-compose -f docker-compose-live-voice.yml up --build -d

# Wait for services to be healthy
Write-Host "`n⏳ Waiting for services to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check service health
Write-Host "`n🔍 Checking service health..." -ForegroundColor Cyan

$services = @(
    @{Name="Redis"; Port="6379"; Container="vorta-live-redis"}
    @{Name="PostgreSQL"; Port="5433"; Container="vorta-live-postgres"}
    @{Name="Inference Engine"; Port="8001"; Container="vorta-inference-engine"}
    @{Name="Ultra AGI Voice"; Port="8888"; Container="vorta-ultra-agi-voice"}
    @{Name="API Gateway"; Port="8000"; Container="vorta-api-gateway"}
    @{Name="Voice Interface"; Port="8501"; Container="vorta-voice-interface"}
)

foreach ($service in $services) {
    $health = docker inspect --format='{{.State.Health.Status}}' $service.Container 2>$null
    if ($health -eq "healthy") {
        Write-Host "✅ $($service.Name) - Healthy" -ForegroundColor Green
    } elseif ($health -eq "starting") {
        Write-Host "🔄 $($service.Name) - Starting..." -ForegroundColor Yellow
    } else {
        Write-Host "❌ $($service.Name) - Unhealthy" -ForegroundColor Red
    }
}

# Show service URLs
Write-Host "`n🌐 VORTA Live Voice Services:" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "🎙️ Voice Interface:     http://localhost:8501" -ForegroundColor Green
Write-Host "🧠 Ultra AGI Voice:     http://localhost:8888" -ForegroundColor Green
Write-Host "⚡ API Gateway:         http://localhost:8000" -ForegroundColor Green
Write-Host "🔬 Inference Engine:    http://localhost:8001" -ForegroundColor Green
Write-Host "📊 Prometheus:          http://localhost:9091" -ForegroundColor Green
Write-Host "📈 Grafana:             http://localhost:3001" -ForegroundColor Green

# Show WebSocket endpoints
Write-Host "`n🔗 WebSocket Endpoints:" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "🎙️ Live Voice:          ws://localhost:8000/ws/live-voice" -ForegroundColor Green
Write-Host "💬 Chat:                ws://localhost:8000/ws/chat" -ForegroundColor Green

# Show logs command
Write-Host "`n📋 Useful Commands:" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host "View all logs:       docker-compose -f docker-compose-live-voice.yml logs -f" -ForegroundColor Yellow
Write-Host "View Voice logs:     docker logs -f vorta-ultra-agi-voice" -ForegroundColor Yellow
Write-Host "Stop services:       docker-compose -f docker-compose-live-voice.yml down" -ForegroundColor Yellow
Write-Host "Restart services:    docker-compose -f docker-compose-live-voice.yml restart" -ForegroundColor Yellow

Write-Host "`n🎯 READY FOR LIVE VOICE CONVERSATION!" -ForegroundColor Green
Write-Host "Open http://localhost:8501 to start talking with VORTA!" -ForegroundColor Green
