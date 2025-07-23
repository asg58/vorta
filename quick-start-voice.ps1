#!/usr/bin/env pwsh

# VORTA Live Voice - Quick Docker Start
Write-Host "🚀 VORTA Live Voice - Quick Start" -ForegroundColor Cyan

# Start the live voice Docker stack
docker-compose -f docker-compose-live-voice.yml up -d

Write-Host "🎙️ Live Voice Interface: http://localhost:8501" -ForegroundColor Green
Write-Host "🧠 Ultra AGI Voice Agent: http://localhost:8888" -ForegroundColor Green
Write-Host "⚡ API Gateway: http://localhost:8000" -ForegroundColor Green
