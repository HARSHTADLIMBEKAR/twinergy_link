# Twinergy Backend Startup Script
Write-Host "Starting Twinergy Backend Server..." -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

cd backend

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Warning: Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv venv
    & "venv\Scripts\Activate.ps1"
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Check if database exists, initialize if not
if (-not (Test-Path "twinergy.db")) {
    Write-Host "Initializing database..." -ForegroundColor Yellow
    python -c "from database import Database; Database()"
}

Write-Host "Starting Flask server on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "=================================" -ForegroundColor Cyan
python app.py


