@echo off
setlocal enabledelayedexpansion

REM --- paths ---
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "BACK=%ROOT%\CarCondition"
set "FRONT=%ROOT%\FrontEnd\car-condition-frontend"

REM --- .env for frontend ---
if not exist "%FRONT%\.env" (
  echo NUXT_PUBLIC_API_BASE=http://localhost:8000>"%FRONT%\.env"
) else (
  findstr /B /I "NUXT_PUBLIC_API_BASE=" "%FRONT%\.env" >nul || (
    echo NUXT_PUBLIC_API_BASE=http://localhost:8000>>"%FRONT%\.env"
  )
)

REM --- .env for backend ---
if not exist "%BACK%\.env" (
  echo Creating backend .env file...
  (
    echo MODEL_PATH=models/model.pth
    echo DEVICE=cpu
    echo # если нет интернета или не хочешь докачивать resnet weights:
    echo RESNET_WEIGHTS=NONE
  ) > "%BACK%\.env"
)

REM --- Python chooser (py -3 preferred) ---
where py >nul 2>&1 && (set "PYCMD=py -3") || (set "PYCMD=python")
echo Using Python launcher: %PYCMD%

REM --- Test Python is working ---
%PYCMD% --version >nul 2>&1
if %errorlevel% neq 0 (
  echo ERROR: Python not found or not working. Please install Python.
  pause
  exit /b 1
)

REM --- venv create (if missing) ---
if not exist "%BACK%\.venv\Scripts\python.exe" (
  echo Creating virtualenv in CarCondition\.venv
  pushd "%BACK%"
  %PYCMD% -m venv .venv
  if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    popd
    pause
    exit /b 1
  )
  popd
)

set "VENV_PY=%BACK%\.venv\Scripts\python.exe"
set "VENV_PIP=%BACK%\.venv\Scripts\pip.exe"

REM --- backend deps ---
echo Upgrading pip and installing backend deps...
"%VENV_PY%" -m pip install --upgrade pip
if %errorlevel% neq 0 (
  echo ERROR: Failed to upgrade pip
  pause
  exit /b 1
)

"%VENV_PIP%" install -r "%BACK%\requirements.txt"
if %errorlevel% neq 0 (
  echo ERROR: Failed to install backend dependencies
  pause
  exit /b 1
)

REM --- start backend in new window ---
echo Starting FastAPI on :8000 ...
start "FastAPI :8000" cmd /k "cd /d "%BACK%" && "%VENV_PY%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM --- frontend deps ---
pushd "%FRONT%"
where pnpm >nul 2>&1
if %errorlevel%==0 (
  echo Installing frontend deps with pnpm...
  pnpm install
  echo Starting Nuxt on :3000 ...
  start "Nuxt :3000" cmd /k "cd /d "%FRONT%" && pnpm dev"
) else (
  echo pnpm not found. Using npm...
  call npm install
  echo Starting Nuxt on :3000 ...
  start "Nuxt :3000" cmd /k "cd /d "%FRONT%" && npm run dev"
)
popd

REM --- open browser ---
start "" http://localhost:3000

echo.
echo ✅ FastAPI и Nuxt запущены в отдельных окнах.
echo Закройте эти окна, чтобы остановить процессы (или используйте stop_all.bat).
echo.
endlocal
