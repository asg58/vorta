# VORTA ULTRA - WORKSPACE CLEANUP SCRIPT (PowerShell)
# Keeps only essential ultra-fast build files

Write-Host "🚀 VORTA ULTRA - Cleaning workspace for MAXIMUM SPEED!" -ForegroundColor Green

# Remove cache directories
Write-Host "Removing cache directories..." -ForegroundColor Yellow
Remove-Item -Recurse -Force .mypy_cache -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Name "__pycache__" | ForEach-Object { 
    Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue 
}

# Remove temporary files
Write-Host "Removing temporary files..." -ForegroundColor Yellow
Remove-Item -Force *.tmp -ErrorAction SilentlyContinue
Remove-Item -Force *.temp -ErrorAction SilentlyContinue
Remove-Item -Force *.log -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force logs -ErrorAction SilentlyContinue

# Remove old Docker build artifacts
Write-Host "Cleaning Docker build cache..." -ForegroundColor Yellow
docker system prune -f --volumes
docker builder prune -f

# Clean pip cache
Write-Host "Cleaning pip cache..." -ForegroundColor Yellow
pip cache purge

# Display current ultra-fast build files
Write-Host "✅ ULTRA-FAST BUILD FILES PRESERVED:" -ForegroundColor Green
Write-Host "🐳 DOCKER FILES:" -ForegroundColor Cyan
Get-ChildItem . -Name "docker-compose.yml", "Dockerfile*", ".dockerignore"
Get-ChildItem frontend -Name "Dockerfile", ".dockerignore", "pip.conf"

Write-Host "⚡ PYTHON OPTIMIZATION FILES:" -ForegroundColor Cyan
Get-ChildItem frontend -Name "requirements.txt", "pip.conf"

Write-Host "🚀 BUILD SCRIPTS:" -ForegroundColor Cyan
Get-ChildItem . -Name "build-ultra-fast.*"

# Calculate space saved
$totalSize = (Get-ChildItem -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "📊 Workspace optimized! Current size: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Green

Write-Host "🎯 WORKSPACE ULTRA-CLEAN FOR MAXIMUM BUILD SPEED! 🎯" -ForegroundColor Green
