@echo off
::set HSA_OVERRIDE_GFX_VERSION=11.5.1
::set CUDA_VISIBLE_DEVICES=0
::set HIP_VISIBLE_DEVICES=1

::set AMD_SERIALIZE_KERNEL=1 -> crash log
::set AMD_LOG_LEVEL=3 -> logging

@echo off
title Cleaner - DEBUG

echo ==========================================
echo       CLEANER STARTUP
echo ==========================================
echo.
echo Project folder:
echo %~dp0
echo.

echo Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

if errorlevel 1 (
    echo.
    echo !!! FAILED TO ACTIVATE VENV !!!
    pause
    exit /b 1
)

echo.
echo Python:
python --version

echo.
echo Python location:
where python

echo.
echo Starting Streamlit...
echo ==========================================
echo.

python -m streamlit run "%~dp0main.py"

echo.
echo ==========================================
echo Streamlit exited.
echo Exit code: %errorlevel%
echo ==========================================
pause