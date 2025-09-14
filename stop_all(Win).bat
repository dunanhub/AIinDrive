@echo off
setlocal

REM kill uvicorn bound to :8000
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":8000 .*LISTENING"') do (
  echo Stopping FastAPI (pid %%p) & taskkill /PID %%p /T /F >nul 2>&1
)

REM kill node/nuxt bound to :3000
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":3000 .*LISTENING"') do (
  echo Stopping Nuxt (pid %%p) & taskkill /PID %%p /T /F >nul 2>&1
)

REM also try by window title (if still alive)
taskkill /FI "WINDOWTITLE eq FastAPI :8000" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Nuxt :3000" /T /F >nul 2>&1

echo ✅ Stopped (where applicable).
endlocal
