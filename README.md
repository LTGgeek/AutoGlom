# AutoGlom Application
## Overview
AutoGlom is an advanced software tool for analyzing and processing kidney MRI (Magnetic Resonance Imaging) data. It automates kidney segmentation and glomeruli detection using deep learning techniques with Python-based image processing.

## Features
- Kidney Boundary Detection
- Kidney Segmentation
- U-Net Based Inference
- U-HDoG Analysis
- Interactive Viewer
- Quantitative Analysis

## Setup and Installation
### 1. Anaconda Environment Setup
1. Install Anaconda or Miniconda
2. Locate `autoglom_app.yml` in the project repository
3. Open an Anaconda prompt and navigate to the directory containing `autoglom_app.yml`
4. Create and activate the environment:
```bash
conda env create -f autoglom_app.yml
conda activate autoglom_app
```

### 2. Verifying the Installation
1. Activate the Anaconda environment:
```bash
conda activate autoglom_app
```
2. Start a Python session:
```bash
python
```
3. Try importing the required libraries:
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
```
If these imports work without errors, your environment is correctly configured.

## Configuration
### Updating config.toml
Modify the `config.toml` file to set appropriate paths:
```toml
[paths]
input_dir = "/path/to/input"
output_dir = "/path/to/output"
model_dir = "/path/to/model"
```

## Running the Application
1. Activate the environment:
```bash
conda activate autoglom_app
```
2. Run the application:
```bash
python demo.py
```

## User Interface and Features
- Initial Setup Window
- Kidney Boundary Check Interface
- Kidney Segmentation Interface
- U-Net Inference Interface
- U-HDoG Inference Interface

## Operation Flow
1. Initial Setup
2. Kidney Boundary Check
3. Kidney Segmentation
4. U-Net Inference
5. U-HDoG Inference

## Image Requirements
- Size: Square, max 256x256 pixels
- File Formats: PNG and JPEG
- Consistency: All images must have the same dimensions

## Troubleshooting
- Module Not Found: Check Anaconda environment activation and dependencies
- CUDA Errors: Check NVIDIA GPU compatibility and CUDA installation

## System Requirements
- Operating System: Windows 10 or later
- NVIDIA GPU: Required for CUDA support
- Dependencies: Installed via `autoglom_app.yml`

## Creating an Installer with PyInstaller
To distribute the application as an executable, use PyInstaller to package `autoglom_app.py` into a standalone application.

### Steps to Create an Installer
1. Install PyInstaller in your environment:
```bash
pip install pyinstaller
```

2. Navigate to the directory containing `autoglom_app.py`:
```bash
cd /path/to/your/project
```

3. Run the PyInstaller command to create an executable:
```bash
pyinstaller --onefile autoglom_app.py
```

4. After the process completes, the standalone executable will be located in the `dist` directory.

### Additional Options
- To prevent a console window from appearing when running the executable:
```bash
pyinstaller --onefile --noconsole autoglom_app.py
```

- To include additional files (like configuration files) in the executable:
```bash
pyinstaller --onefile --add-data "config.toml;." autoglom_app.py
```

Note: Ensure all necessary dependencies and files are available during runtime. You may need to include additional files or folders using the `--add-data` option.
