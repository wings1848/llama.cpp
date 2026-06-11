@echo off
REM ============================================================
REM build_swlp_cuda.bat - SWLP CUDA build for Windows
REM
REM Usage:
REM   First time:  build_swlp_cuda.bat
REM   Incremental: build_swlp_cuda.bat
REM   Full rebuild: build_swlp_cuda.bat --clean
REM
REM Auto-detects vcvars64.bat across VS 2022 editions.
REM Set VCVARS64_BAT env var to override.
REM ============================================================
setlocal enabledelayedexpansion

set BUILD_DIR=build_cuda_swlp

REM Auto-detect vcvars64.bat
if not defined VCVARS64_BAT (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64_BAT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64_BAT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64_BAT=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64_BAT=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if not defined VCVARS64_BAT (
    echo [ERROR] Cannot find vcvars64.bat.
    echo   Set VCVARS64_BAT env var or install Visual Studio 2022.
    exit /b 1
)

REM Check if we need a clean build
if "%1"=="--clean" (
    echo [CLEAN] Removing %BUILD_DIR%...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
)
if "%1"=="-c" (
    echo [CLEAN] Removing %BUILD_DIR%...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
)

REM Load MSVC environment
echo [ENV] Loading MSVC build tools...
call "%VCVARS64_BAT%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed
    echo   Path: "%VCVARS64_BAT%"
    exit /b 1
)

REM Decide: configure or incremental?
set NEED_CONFIGURE=0
if not exist "%BUILD_DIR%\CMakeCache.txt" set NEED_CONFIGURE=1

if %NEED_CONFIGURE% EQU 1 (
    echo.
    echo ============================================================
    echo [CMake] First-time configure...
    echo ============================================================
    cmake -B %BUILD_DIR% -G Ninja ^
        -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
        -DCMAKE_C_COMPILER=cl ^
        -DCMAKE_CXX_COMPILER=cl ^
        -DGGML_CUDA=ON ^
        -DBUILD_SHARED_LIBS=ON ^
        -DGGML_STATIC=OFF
    if errorlevel 1 (
        echo [ERROR] CMake configure failed
        exit /b 1
    )
) else (
    echo [CMake] Using existing configuration in %BUILD_DIR% (incremental)
)

echo.
echo ============================================================
echo [BUILD] Building llama-swlp-bench...
echo ============================================================
cmake --build %BUILD_DIR% --config RelWithDebInfo --target llama-swlp-bench
if errorlevel 1 (
    echo [ERROR] Build failed
    exit /b 1
)

echo.
echo ============================================================
echo [OK] Build successful!
echo   Binary: %BUILD_DIR%\bin\llama-swlp-bench.exe
echo ============================================================
exit /b 0
