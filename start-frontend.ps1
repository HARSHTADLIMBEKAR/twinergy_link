# Twinergy Frontend Startup Script
Write-Host "Starting Twinergy Frontend..." -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

cd frontend

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "Starting React development server..." -ForegroundColor Green
Write-Host "Frontend will open at http://localhost:3000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "=============================" -ForegroundColor Cyan
npm start


