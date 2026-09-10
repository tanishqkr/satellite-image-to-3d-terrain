@echo off
echo.
echo  ============================================
echo   DepthWizard - Single-View Height Estimation
echo   SIH26175 - ISRO SAC
echo  ============================================
echo.
echo Starting backend server...
start /B cmd /c "cd backend && ..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo Backend starting on http://localhost:8000
echo.
timeout /t 5 /nobreak >nul
echo Starting frontend...
cd frontend && npm run dev
