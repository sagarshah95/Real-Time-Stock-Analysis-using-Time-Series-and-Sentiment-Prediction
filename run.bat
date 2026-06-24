@echo off
cd /d "%~dp0"
echo Installing dependencies if needed...
python -m pip install -r requirements.txt -q
echo Starting F.A.S.T Streamlit app...
python -m streamlit run app.py
pause
