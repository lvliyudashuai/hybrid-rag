@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

echo ============================================
echo  RAG Demo Launcher
echo ============================================
echo Starting Streamlit at http://localhost:8501 ...
echo Close this window to stop it.
"%PY%" -E -X utf8 -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1

echo.
echo If the browser did not open, visit http://localhost:8501
pause
