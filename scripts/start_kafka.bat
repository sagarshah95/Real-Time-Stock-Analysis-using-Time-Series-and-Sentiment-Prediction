@echo off
cd /d "%~dp0\.."
echo Starting Kafka + Zookeeper...
docker compose -f infra\docker-compose.kafka.yml up -d
if errorlevel 1 (
    echo.
    echo Failed. Is Docker Desktop running?
    pause
    exit /b 1
)
echo.
echo Waiting for broker on localhost:9092...
timeout /t 12 /nobreak >nul
docker compose -f infra\docker-compose.kafka.yml ps
echo.
echo Kafka should be ready. Restart Streamlit or click Reconnect Kafka on Social trends.
pause
