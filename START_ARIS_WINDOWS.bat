\
@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Devoniq ARIS Open Core - Simulation Lab

echo ============================================================
echo   DEVONIQ ARIS OPEN CORE - SIMULATION LAB
echo ============================================================
echo.
echo Generic Bullet rigid-body simulation. No real aircraft required.
echo.

set "PY_CMD="
for %%V in (3.11 3.12 3.13) do (
  if not defined PY_CMD (
    py -%%V -c "import sys; print(sys.version)" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -%%V"
  )
)

if not defined PY_CMD (
  python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,13)" >nul 2>&1
  if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD goto no_python

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating virtual environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto failed
) else (
  echo [1/4] Virtual environment exists.
)

set "VPY=%CD%\.venv\Scripts\python.exe"

echo [2/4] Installing/checking dependencies...
"%VPY%" -c "import fastapi,uvicorn,pydantic,httpx,PIL,numpy,cv2,shapely,pybullet" >nul 2>&1
if errorlevel 1 (
  "%VPY%" -m pip install --upgrade pip
  if errorlevel 1 goto failed
  "%VPY%" -m pip install -r requirements-sim.txt
  if errorlevel 1 goto dependency_failed
)

echo [3/4] Running community self-test...
"%VPY%" -m tools.community_self_test
if errorlevel 1 goto smoke_failed

echo [4/4] Starting Simulation Lab...
echo.
echo Live natural-language interpretation is optional and requires:
echo   OPENAI_API_KEY
echo   OPENAI_MISSION_MODEL
echo.
"%VPY%" -m simulation.launch
if errorlevel 1 goto failed
exit /b 0

:no_python
echo ERROR: Python 3.11-3.13 was not found.
echo Recommended: install Python 3.11 x64 from python.org.
pause
exit /b 1

:dependency_failed
echo ERROR: Python dependencies could not be installed.
echo Make sure this PC has Internet access and retry.
pause
exit /b 1

:smoke_failed
echo ERROR: ARIS self-test failed. Copy the error above into a GitHub issue.
pause
exit /b 1

:failed
echo ERROR: ARIS setup or launch failed.
pause
exit /b 1
