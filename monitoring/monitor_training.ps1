# Script para monitorear el entrenamiento
Write-Host "🔍 MONITOREANDO ENTRENAMIENTO" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Activar entorno virtual
& "c:/Users/Alex B/Desktop/bot_trading_v9/bot_trading_v9.1/venv/Scripts/Activate.ps1"

# Verificar estado
Write-Host "📊 Verificando estado..." -ForegroundColor Yellow
python check_status.py

Write-Host "`n🔍 Monitoreando fixes..." -ForegroundColor Yellow
python monitor_fixes.py
