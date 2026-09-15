@echo off
chcp 65001 >nul
title RAG 文档问答助手
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

rem 找一个能用的 Python：优先项目自带虚拟环境，其次本机 3.12，最后退回 PATH
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if defined PY goto run
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto run
set "PY=python"

:run
%PY% -E launch.py
if errorlevel 1 pause
