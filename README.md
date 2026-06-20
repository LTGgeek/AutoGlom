# AutoGlom

AutoGlom is a Windows application for kidney MRI processing and glomerular segmentation. It converts NRRD image/annotation volumes into PNG slices, prepares kidney/cortex masks, runs U-Net inference, performs UHDoG glomerular analysis, and reports volume measurements.

## Environment Setup

AutoGlom is developed for Windows 64-bit with Anaconda or Miniconda. The recommended environment for packaging and distribution is the CPU environment because it does not require users to install CUDA or cuDNN.

### Recommended CPU Environment

Open an **Anaconda Prompt**, go to this folder, and run:

```powershell
setup_autoglom_cpu_env.bat
```

The script will:

- Create or update the `autoglom_app_cpu` conda environment from `autoglom_cpu.yml`.
- Install `PyInstaller==5.13.2`.
- Verify the required packages can be imported.

The CPU environment includes:

- Python 3.9
- TensorFlow 2.6.0
- Keras 2.6.0
- NumPy, SciPy, scikit-image
- OpenCV
- Pillow
- matplotlib
- pynrrd
- PyInstaller

Manual setup commands:

```powershell
conda env create -f autoglom_cpu.yml
conda run -n autoglom_app_cpu python -m pip install pyinstaller==5.13.2
conda run -n autoglom_app_cpu python -c "import tensorflow, keras, numpy, scipy, skimage, cv2, nrrd, PyInstaller; print('cpu build environment ok')"
```

If `autoglom_app_cpu` already exists, update it instead:

```powershell
conda env update -n autoglom_app_cpu -f autoglom_cpu.yml --prune
```

### Original GPU/Development Environment

The original source workflow uses the `kidney` conda environment on this workstation:

```powershell
conda activate kidney
python autoglom_app1.py
```

For a reproducible GPU-style environment, use:

```powershell
conda env create -f autoglom_app_python.yml
```

This GPU-style environment includes CUDA/cuDNN dependencies. It is useful for the original GPU build, but it is less portable than the CPU executable.

## Main Entry Point

Run the source application with:

```powershell
conda activate kidney
python autoglom_app1.py
```

If you created the GPU-style environment from `autoglom_app_python.yml`, activate `autoglom_app` instead of `kidney`.

The GUI starts with a setup window where you select:

- Image NRRD file
- Kidney annotation NRRD file
- Optional medulla annotation NRRD file
- Optional artifact/black-dot annotation NRRD file
- Sector/sample ID
- Output folder
- Slice axis

## Packaged Executables

Prebuilt executables are in:

```text
dist\
```

Available builds:

- `autoglom_app1.exe`: original GPU-style build from the `kidney` environment.
- `autoglom_app1_cpu.exe`: CPU-focused build that does not require CUDA/cuDNN installation.

For most Windows computers, use:

```text
dist\autoglom_app1_cpu.exe
```

The CPU exe uses PyInstaller runtime hook `pyi_runtime_cpu.py` to set CPU mode before TensorFlow starts. This avoids changing the application source code.

## Workflow

1. Open `autoglom_app1.py` or `dist\autoglom_app1_cpu.exe`.
2. Select the NRRD files and output folder.
3. Click `Create Directory Structure`.
4. In the viewer, run `Run Deep Learning Inference`.
5. Run `Glomerular Analysis`.
6. Review the image overlays, histogram popup, and saved result files.

## Output Files

For a sector/sample named `1326`, analysis outputs are written under:

```text
1326_results\
```

Important files:

- `results.json`: summary measurements, including kidney volume, medulla volume, glomerular count, mean glomerular volume, and median glomerular volume.
- `glomerular_volumes.csv`: per-glomerulus volume table.
- `vs_histogram.png`: histogram of positive glomerular volumes.
- Numbered PNG files: UHDoG overlay/result slices.

`glomerular_volumes.csv` contains:

```csv
glom_id,volume_mm3
```

Rows with `volume_mm3 == 0` are skipped.

## Rebuilding The EXE

### CPU Build

Use this build when distributing to computers that may not have CUDA/cuDNN installed:

```powershell
conda run -n autoglom_app_cpu pyinstaller --clean --noconfirm autoglom_app1_cpu.spec
```

Output:

```text
dist\autoglom_app1_cpu.exe
```

The CPU package uses `pyi_runtime_cpu.py` through `autoglom_app1_cpu.spec` to request CPU execution before TensorFlow starts.

### Original GPU-Environment Build

```powershell
conda run -n kidney pyinstaller --clean --noconfirm autoglom_app1.spec
```

Output:

```text
dist\autoglom_app1.exe
```

## Required Model

The U-Net model file must remain in the project folder:

```text
kidney_427.hdf5
```

Both PyInstaller spec files include this model in the executable.

## Notes

- The CPU executable is large because it bundles Python, TensorFlow, scientific Python libraries, and the model.
- The CPU executable may still print TensorFlow CUDA-related messages on some systems, but it is intended to run without installing CUDA or cuDNN.
- `build/` and `__pycache__/` are generated folders and are not needed for normal use.
