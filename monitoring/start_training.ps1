# Script para iniciar el entrenamiento con el entorno virtual activado
Write-Host "🚀 INICIANDO ENTRENAMIENTO DEL BOT" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Activar entorno virtual y ejecutar entrenamiento
Write-Host "📦 Activando entorno virtual y ejecutando entrenamiento..." -ForegroundColor Yellow
& "c:/Users/Alex B/Desktop/bot_trading_v9/bot_trading_v9.1/venv/Scripts/Activate.ps1"; python app.py run

Write-Host "✅ Entrenamiento completado" -ForegroundColor Green
