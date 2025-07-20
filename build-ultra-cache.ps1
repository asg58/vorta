# 🚀 VORTA ULTRA CACHE BUILD SCRIPT
# Ultra-fast build met pip cache mount optimization

Write-Host "🚀 VORTA ULTRA - Building with ADVANCED CACHE OPTIMIZATION!" -ForegroundColor Green

# Set ultra-performance environment
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1
$env:BUILDKIT_INLINE_CACHE=1

Write-Host "🔧 Environment Variables:" -ForegroundColor Cyan
Write-Host "DOCKER_BUILDKIT: $($env:DOCKER_BUILDKIT)" -ForegroundColor Yellow
Write-Host "COMPOSE_DOCKER_CLI_BUILD: $($env:COMPOSE_DOCKER_CLI_BUILD)" -ForegroundColor Yellow
Write-Host "BUILDKIT_INLINE_CACHE: $($env:BUILDKIT_INLINE_CACHE)" -ForegroundColor Yellow

Write-Host "" 
Write-Host "📦 Building with PIP CACHE MOUNT for ultra-fast PyTorch downloads..." -ForegroundColor Green

# Build only API service with advanced cache mount (ultra-fast!)
docker-compose build vorta-api

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "✅ ULTRA BUILD SUCCESS! PyTorch and large packages cached!" -ForegroundColor Green
    Write-Host "🚀 Next builds will be lightning fast!" -ForegroundColor Cyan
} else {
    Write-Host "❌ Build failed with exit code: $exitCode" -ForegroundColor Red
}
