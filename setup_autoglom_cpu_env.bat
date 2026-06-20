@echo off
setlocal

echo Setting up AutoGlom CPU build environment...

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found. Open an Anaconda Prompt or install Anaconda/Miniconda first.
    exit /b 1
)

conda env list | findstr /R /C:"^autoglom_app_cpu " >nul
if errorlevel 1 (
    echo Creating conda environment autoglom_app_cpu from autoglom_cpu.yml...
    conda env create -f autoglom_cpu.yml
) else (
    echo Updating existing conda environment autoglom_app_cpu from autoglom_cpu.yml...
    conda env update -n autoglom_app_cpu -f autoglom_cpu.yml --prune
)

if errorlevel 1 (
    echo ERROR: conda environment setup failed.
    exit /b 1
)

echo Installing PyInstaller for exe packaging...
conda run -n autoglom_app_cpu python -m pip install pyinstaller==5.13.2

if errorlevel 1 (
    echo ERROR: PyInstaller installation failed.
    exit /b 1
)

echo Verifying required Python packages...
conda run -n autoglom_app_cpu python -c "import tensorflow, keras, numpy, scipy, skimage, cv2, nrrd, PyInstaller; print('AutoGlom CPU environment is ready')"

if errorlevel 1 (
    echo ERROR: Environment verification failed.
    exit /b 1
)

echo Done. To build the CPU exe, run:
echo conda run -n autoglom_app_cpu pyinstaller --clean --noconfirm autoglom_app1_cpu.spec

endlocal
