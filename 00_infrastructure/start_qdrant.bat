@echo off
TITLE Qdrant Database Launcher for AMD RAG
ECHO ===================================================
ECHO      Starting Qdrant Vector Database...
ECHO ===================================================

:: 1. Check if Docker is running
docker info >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Docker is NOT running! 
    ECHO Please start Docker Desktop first and try again.
    PAUSE
    EXIT /B
)

:: 2. Check if container already exists
docker ps -a --format '{{.Names}}' | findstr "^qdrant_amd_rag$" >nul
IF %ERRORLEVEL% EQU 0 (
    ECHO [INFO] Container 'qdrant_amd_rag' found. Starting it...
    docker start qdrant_amd_rag
) ELSE (
    ECHO [INFO] Creating new Qdrant container...
    :: Pulling the image first to ensure we have it
    docker pull qdrant/qdrant
    docker run -d -p 6333:6333 -p 6334:6334 --name qdrant_amd_rag qdrant/qdrant
)

ECHO.
ECHO [SUCCESS] Qdrant is running!
ECHO Dashboard available at: http://localhost:6333/dashboard
ECHO.
ECHO You can now close this window and proceed to Step 1 (Python Setup).
PAUSE
