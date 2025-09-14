@echo off
setlocal enabledelayedexpansion

echo Testing start_all script without launching services...
echo.

REM --- paths ---
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "BACK=%ROOT%\CarCondition"
set "FRONT=%ROOT%\FrontEnd\car-condition-frontend"

echo Paths:
echo   ROOT: %ROOT%
echo   BACK: %BACK%
echo   FRONT: %FRONT%
echo.

REM --- Test directories exist ---
if not exist "%BACK%" (
  echo ERROR: Backend directory not found: %BACK%
  pause
  exit /b 1
)

if not exist "%FRONT%" (
  echo ERROR: Frontend directory not found: %FRONT%
  pause
  exit /b 1
)

echo ✓ Directories exist
echo.

REM --- Test Python ---
where py >nul 2>&1 && (set "PYCMD=py -3") || (set "PYCMD=python")
echo Testing Python: %PYCMD%
%PYCMD% --version
if %errorlevel% neq 0 (
  echo ERROR: Python not working
  pause
  exit /b 1
)
echo ✓ Python working
echo.

REM --- Test venv ---
if exist "%BACK%\.venv\Scripts\python.exe" (
  echo ✓ Virtual environment exists
  "%BACK%\.venv\Scripts\python.exe" --version
) else (
  echo ! Virtual environment not found (will be created)
)
echo.

REM --- Test package managers ---
where pnpm >nul 2>&1
if %errorlevel%==0 (
  echo ✓ pnpm found
  pnpm --version
) else (
  where npm >nul 2>&1
  if %errorlevel%==0 (
    echo ✓ npm found ^(pnpm not available^)
    npm --version
  ) else (
    echo ERROR: Neither pnpm nor npm found
    pause
    exit /b 1
  )
)

echo.
echo ✅ All tests passed! The start_all script should work.
pause