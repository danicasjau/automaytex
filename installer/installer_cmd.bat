@echo off
setlocal EnableDelayedExpansion
title AutoMayTex - Maya Plugin Installer
color 0A

echo.
echo  =====================================================
echo   AutoMayTex - Maya Plugin Installer
echo  =====================================================
echo.

:: ─────────────────────────────────────────────────────
:: STEP 1 — SELECT MAYA VERSION
:: ─────────────────────────────────────────────────────
echo  [STEP 1/7] Select your Maya installation...
echo.

:: Scan common Maya install locations
set "MAYA_BASE=C:\Program Files\Autodesk"
set MAYA_COUNT=0

if exist "%MAYA_BASE%" (
    for /d %%D in ("%MAYA_BASE%\Maya*") do (
        set /a MAYA_COUNT+=1
        set "MAYA_OPT_!MAYA_COUNT!=%%D"
        set "MAYA_LABEL_!MAYA_COUNT!=%%~nD"
    )
)

if %MAYA_COUNT%==0 (
    echo  [WARNING] No Maya installations found in "%MAYA_BASE%".
    echo  Please enter the full path to your Maya installation manually:
    set /p "MAYA_DIR=  Maya path: "
    if not exist "!MAYA_DIR!" (
        echo  [ERROR] Path does not exist. Exiting.
        pause & exit /b 1
    )
    goto :maya_selected
)

echo  Found the following Maya installations:
echo.
for /l %%i in (1,1,%MAYA_COUNT%) do (
    echo    [%%i] !MAYA_LABEL_%%i!
)
echo    [M] Enter path manually
echo.
set /p "MAYA_CHOICE=  Select Maya version (number): "

if /i "%MAYA_CHOICE%"=="M" (
    set /p "MAYA_DIR=  Enter full Maya path: "
) else (
    set "MAYA_DIR=!MAYA_OPT_%MAYA_CHOICE%!"
)

if not defined MAYA_DIR (
    echo  [ERROR] Invalid selection. Exiting.
    pause & exit /b 1
)
if not exist "!MAYA_DIR!" (
    echo  [ERROR] Maya directory not found: !MAYA_DIR!
    pause & exit /b 1
)

:maya_selected
echo.
echo  [OK] Maya directory: !MAYA_DIR!

:: ─────────────────────────────────────────────────────
:: STEP 2 — FIND MAYA PYTHON
:: ─────────────────────────────────────────────────────
echo.
echo  [STEP 2/7] Locating Maya Python executable...
echo.

:: Maya ships mayapy.exe inside bin\
set "MAYAPY=!MAYA_DIR!\bin\mayapy.exe"

if not exist "!MAYAPY!" (
    echo  [ERROR] mayapy.exe not found at expected location:
    echo          !MAYAPY!
    echo  Please enter the full path to mayapy.exe manually:
    set /p "MAYAPY=  mayapy.exe path: "
    if not exist "!MAYAPY!" (
        echo  [ERROR] mayapy.exe not found. Exiting.
        pause & exit /b 1
    )
)

:: Detect Python version from mayapy
for /f "tokens=2 delims= " %%V in ('"!MAYAPY!" --version 2^>^&1') do set "PYTHON_VER=%%V"
for /f "tokens=1,2 delims=." %%A in ("!PYTHON_VER!") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)
set "PY_SHORT=python!PY_MAJOR!.!PY_MINOR!"

echo  [OK] Found: !MAYAPY!
echo  [OK] Python version: !PYTHON_VER! (!PY_SHORT!)

:: Validate supported version
if "!PY_MAJOR!!PY_MINOR!"=="310" goto :pyver_ok
if "!PY_MAJOR!!PY_MINOR!"=="311" goto :pyver_ok
if "!PY_MAJOR!!PY_MINOR!"=="312" goto :pyver_ok
echo  [WARNING] Python !PYTHON_VER! is not explicitly supported (expected 3.10 / 3.11 / 3.12).
echo  Continuing anyway — requirements.txt selection may default to 3.12.
:pyver_ok

:: ─────────────────────────────────────────────────────
:: STEP 3 — DEFAULT PLUGIN INSTALLATION PATH
:: ─────────────────────────────────────────────────────
echo.
echo  [STEP 3/7] Plugin installation path...
echo.

:: Derive a sensible default from the Maya user prefs dir
for /f "usebackq tokens=*" %%H in (`echo %USERPROFILE%`) do set "UP=%%H"
set "DEFAULT_INSTALL_PATH=%UP%\Documents\maya\AutoMayTex"

set /p "INSTALL_PATH=  Installation path [!DEFAULT_INSTALL_PATH!]: "
if "!INSTALL_PATH!"=="" set "INSTALL_PATH=!DEFAULT_INSTALL_PATH!"

echo.
echo  [OK] Installation path: !INSTALL_PATH!

:: Create directory if it doesn't exist
if not exist "!INSTALL_PATH!" (
    mkdir "!INSTALL_PATH!" 2>nul
    if errorlevel 1 (
        echo  [ERROR] Could not create installation directory. Check permissions.
        pause & exit /b 1
    )
    echo  [OK] Created directory: !INSTALL_PATH!
)

:: ─────────────────────────────────────────────────────
:: STEP 4 — GIT CLONE REPOSITORY
:: ─────────────────────────────────────────────────────
echo.
echo  [STEP 4/7] Repository setup...
echo.

:: Check git is available
where git >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Git is not installed or not on PATH.
    echo  Please install Git for Windows: https://git-scm.com/download/win
    pause & exit /b 1
)

set "DEFAULT_REPO=https://github.com/danicasjau/automaytex.git"
set /p "REPO_URL=  Enter repository URL [!DEFAULT_REPO!]: "
if "!REPO_URL!"=="" set "REPO_URL=!DEFAULT_REPO!"

echo.
echo  Cloning !REPO_URL! into !INSTALL_PATH! ...
echo.

git clone "!REPO_URL!" "!INSTALL_PATH!" 2>&1
if errorlevel 1 (
    :: If folder already exists with repo, try a pull instead
    if exist "!INSTALL_PATH!\.git" (
        echo  [INFO] Directory already contains a git repo. Pulling latest changes...
        cd /d "!INSTALL_PATH!"
        git pull 2>&1
        if errorlevel 1 (
            echo  [ERROR] git pull failed. Please check your connection and permissions.
            pause & exit /b 1
        )
    ) else (
        echo  [ERROR] git clone failed. Check the repository URL and your internet connection.
        pause & exit /b 1
    )
)

echo.
echo  [OK] Repository ready at: !INSTALL_PATH!

:: ─────────────────────────────────────────────────────
:: STEP 5 — OPTIONAL MODEL INSTALLATION
:: ─────────────────────────────────────────────────────
echo.
echo  [STEP 5/7] Model installation...
echo.

choice /c YN /m "  Do you want to install models? (Y/N)"
if errorlevel 2 goto :skip_models
if errorlevel 1 goto :install_models

:install_models
set "MODEL_SCRIPT=!INSTALL_PATH!\model_installation.py"
if not exist "!MODEL_SCRIPT!" (
    echo  [WARNING] model_installation.py not found at:
    echo           !MODEL_SCRIPT!
    set /p "MODEL_SCRIPT=  Enter full path to model_installation.py: "
    if not exist "!MODEL_SCRIPT!" (
        echo  [ERROR] Script not found. Skipping model installation.
        goto :skip_models
    )
)
echo.
echo  Running model installation script...
"!MAYAPY!" "!MODEL_SCRIPT!"
if errorlevel 1 (
    echo  [WARNING] model_installation.py exited with errors. Continuing...
) else (
    echo  [OK] Models installed successfully.
)

:skip_models

:: ─────────────────────────────────────────────────────
:: STEP 6 — CREATE VENV + INSTALL REQUIREMENTS
:: ─────────────────────────────────────────────────────
echo.
echo  [STEP 6/7] Creating Python environment and installing dependencies...
echo.

:: Check CUDA availability
set "CUDA_AVAILABLE=0"
where nvcc >nul 2>&1 && set "CUDA_AVAILABLE=1"
if "!CUDA_AVAILABLE!"=="0" (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" set "CUDA_AVAILABLE=1"
)

if "!CUDA_AVAILABLE!"=="1" (
    echo  [OK] CUDA detected — GPU-accelerated packages will be preferred.
) else (
    echo  [INFO] CUDA not detected — CPU-only packages will be used.
)

:: Select correct requirements file based on Python version
if "!PY_MAJOR!!PY_MINOR!"=="310" (
    if "!CUDA_AVAILABLE!"=="1" (
        set "REQ_FILE=!INSTALL_PATH!\requirements_py310_cuda.txt"
    ) else (
        set "REQ_FILE=!INSTALL_PATH!\requirements_py310.txt"
    )
    set "REQ_LABEL=Python 3.10"
) else if "!PY_MAJOR!!PY_MINOR!"=="311" (
    if "!CUDA_AVAILABLE!"=="1" (
        set "REQ_FILE=!INSTALL_PATH!\requirements_py311_cuda.txt"
    ) else (
        set "REQ_FILE=!INSTALL_PATH!\requirements_py311.txt"
    )
    set "REQ_LABEL=Python 3.11"
) else (
    if "!CUDA_AVAILABLE!"=="1" (
        set "REQ_FILE=!INSTALL_PATH!\requirements_py312_cuda.txt"
    ) else (
        set "REQ_FILE=!INSTALL_PATH!\requirements_py312.txt"
    )
    set "REQ_LABEL=Python 3.12 (default)"
)

:: Fallback: if CUDA-specific file not found, try the generic one
if not exist "!REQ_FILE!" (
    set "REQ_FILE_BASE=!REQ_FILE!"
    if "!PY_MAJOR!!PY_MINOR!"=="310" set "REQ_FILE=!INSTALL_PATH!\requirements_py310.txt"
    if "!PY_MAJOR!!PY_MINOR!"=="311" set "REQ_FILE=!INSTALL_PATH!\requirements_py311.txt"
    if not "!REQ_FILE_BASE!"=="!REQ_FILE!" (
        if not exist "!REQ_FILE!" set "REQ_FILE=!INSTALL_PATH!\requirements.txt"
    ) else (
        set "REQ_FILE=!INSTALL_PATH!\requirements.txt"
    )
)

if not exist "!REQ_FILE!" (
    echo  [ERROR] Requirements file not found: !REQ_FILE!
    echo  Please ensure requirements files exist in the cloned repository.
    pause & exit /b 1
)

echo  [OK] Using requirements: !REQ_FILE! (!REQ_LABEL!)
echo.

:: Create virtual environment using mayapy
set "VENV_DIR=!INSTALL_PATH!\venv"
if not exist "!VENV_DIR!" (
    echo  Creating virtual environment with mayapy...
    "!MAYAPY!" -m venv "!VENV_DIR!"
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    echo  [OK] Virtual environment created: !VENV_DIR!
) else (
    echo  [INFO] Virtual environment already exists: !VENV_DIR!
)

:: Activate venv and upgrade pip
set "VENV_PIP=!VENV_DIR!\Scripts\pip.exe"
set "VENV_PYTHON=!VENV_DIR!\Scripts\python.exe"

echo.
echo  Upgrading pip...
"!VENV_PYTHON!" -m pip install --upgrade pip --quiet

echo  Installing dependencies from !REQ_FILE! ...
echo  (This may take several minutes depending on package sizes)
echo.
"!VENV_PIP!" install -r "!REQ_FILE!"
if errorlevel 1 (
    echo.
    echo  [ERROR] pip install failed. Check the requirements file and your internet connection.
    pause & exit /b 1
)

echo.
echo  [OK] Dependencies installed successfully.

:: ─────────────────────────────────────────────────────
:: STEP 7 — REGISTER PLUGIN IN MAYA PLUGIN PATH
:: ─────────────────────────────────────────────────────
echo.
echo  [STEP 7/7] Registering AutoMayTex plugin in Maya...
echo.

:: Maya looks for plug-ins in the MAYA_PLUG_IN_PATH env var,
:: or in Documents\maya\<ver>\plug-ins by default.
:: Detect Maya version number from the dir name (e.g. "Maya2025" -> "2025")
for %%D in ("!MAYA_DIR!") do set "MAYA_DIRNAME=%%~nD"
set "MAYA_VER_NUM=!MAYA_DIRNAME:Maya=!"

set "MAYA_PLUGINS_DIR=%USERPROFILE%\Documents\maya\!MAYA_VER_NUM!\plug-ins"
if not exist "!MAYA_PLUGINS_DIR!" (
    mkdir "!MAYA_PLUGINS_DIR!" 2>nul
    echo  [OK] Created Maya plug-ins directory: !MAYA_PLUGINS_DIR!
)

:: Plugin script inside the cloned repo
set "PLUGIN_SRC=!INSTALL_PATH!\automayatex.py"
if not exist "!PLUGIN_SRC!" (
    echo  [WARNING] automayatex.py not found at: !PLUGIN_SRC!
    set /p "PLUGIN_SRC=  Enter full path to automayatex.py: "
    if not exist "!PLUGIN_SRC!" (
        echo  [ERROR] Plugin script not found. Skipping plugin registration.
        goto :done
    )
)

:: Copy (or overwrite) plugin to Maya plug-ins folder
copy /Y "!PLUGIN_SRC!" "!MAYA_PLUGINS_DIR!\automayatex.py" >nul
if errorlevel 1 (
    echo  [ERROR] Failed to copy plugin to: !MAYA_PLUGINS_DIR!
    echo  Try running this script as Administrator.
    pause & exit /b 1
)

echo  [OK] Plugin installed to: !MAYA_PLUGINS_DIR!\automayatex.py

:: Also append the install path to MAYA_PLUG_IN_PATH in the user environment
:: so Maya can also find it directly from the repo folder.
for /f "tokens=3*" %%A in ('reg query "HKCU\Environment" /v MAYA_PLUG_IN_PATH 2^>nul') do set "EXISTING_MPIP=%%A %%B"
if defined EXISTING_MPIP (
    :: Avoid duplicate entries
    echo !EXISTING_MPIP! | findstr /i /c:"!MAYA_PLUGINS_DIR!" >nul 2>&1
    if errorlevel 1 (
        setx MAYA_PLUG_IN_PATH "!EXISTING_MPIP!;!MAYA_PLUGINS_DIR!" >nul
    )
) else (
    setx MAYA_PLUG_IN_PATH "!MAYA_PLUGINS_DIR!" >nul
)

echo  [OK] MAYA_PLUG_IN_PATH updated in user environment.

:done
:: ─────────────────────────────────────────────────────
:: SUMMARY
:: ─────────────────────────────────────────────────────
echo.
echo  =====================================================
echo   Installation Complete!
echo  =====================================================
echo.
echo   Maya dir      : !MAYA_DIR!
echo   Maya Python   : !MAYAPY! (!PYTHON_VER!)
echo   Install path  : !INSTALL_PATH!
echo   CUDA detected : !CUDA_AVAILABLE!
echo   Requirements  : !REQ_FILE!
echo   Venv          : !VENV_DIR!
echo   Plugin dir    : !MAYA_PLUGINS_DIR!
echo.
echo   Next steps:
echo    1. Open Maya
echo    2. Go to Windows ^> Settings/Preferences ^> Plug-in Manager
echo    3. Find automayatex.py and enable "Loaded" and "Auto load"
echo.
echo  =====================================================
echo.
pause
endlocal
exit /b 0