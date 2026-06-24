@echo off
cd /d "%~dp0.."
echo Starting F.A.S.T API on http://localhost:8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
