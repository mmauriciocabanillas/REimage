# REimage

REimage is a local Python runner for CodeFormer. It is intended for users who want to process images on their own machine without queue systems, remote timeouts, paid cloud platforms, or artificial batch limits. Actual speed and stability depend on your hardware, drivers, Python environment, and available GPU memory.

## Goal

The main goal of REimage is to make CodeFormer easier to run in a real local Python environment.

Instead of depending on online demos, limited web interfaces, queues, upload restrictions, or paid cloud services, REimage provides a simple local workflow that users can run directly on their own computer. The project reuses the original CodeFormer technology, but adds a practical Python-based execution structure focused on local use, batch processing, and reproducibility.

In simple terms, REimage exists to make CodeFormer easier to install, execute, and use locally for real image restoration workflows.

## Based on CodeFormer

This repository is not the original CodeFormer model, and the original CodeFormer project was not created by me. The original project belongs to its authors, including sczhou and contributors:

https://github.com/sczhou/CodeFormer

The original CodeFormer source code, model files, and related assets keep their original license and restrictions. REimage only adds a local runner/wrapper around that project.

## What Is Included

* A local `main.py` runner.
* The original `CodeFormer/` project folder.
* Installation notes for a local Python environment.
* Empty tracked folders for user uploads and generated results.
* A simple workflow for running restoration locally.

## What Is Not Included

REimage intentionally does not include:

* Personal user images.
* Generated results.
* Virtual environments.
* ZIP, RAR, 7z, TAR, or GZ archives.
* Model weights, checkpoints, `.pth`, `.pt`, `.ckpt`, or `.safetensors` files.

These files are ignored by Git because they are private, generated, environment-specific, or too large for a clean public repository.

## License Warning

CodeFormer has its own license and may include non-commercial restrictions. Do not assume REimage makes the original project commercially usable. Review the original CodeFormer license before using, modifying, or distributing this project, especially for commercial work.

## Requirements

Python 3.11 is recommended.

For GPU acceleration on Windows, the installation commands use PyTorch with CUDA 12.1. If your machine does not support that setup, install the PyTorch build appropriate for your hardware.

## Installation

Open a terminal in the repository root and follow the commands in:

```txt
install_commands.txt
```

The important setup sequence is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4
python -m pip install basicsr==1.4.2 --no-deps
python -m pip install facexlib gfpgan realesrgan addict future lmdb opencv-python Pillow pyyaml requests scikit-image scipy tqdm yapf lpips gdown
python -m pip install numpy==1.26.4 --force-reinstall
```

## Usage

Put your input images here:

```txt
CodeFormer/inputs/user_upload/
```

Run:

```powershell
python main.py
```

Generated results are saved here:

```txt
CodeFormer/results/
```

## Project Structure

```txt
REimage/
|-- CodeFormer/
|   |-- basicsr/
|   |-- facelib/
|   |-- inference_codeformer.py
|   |-- requirements.txt
|   |-- LICENSE
|   |-- inputs/
|   |   `-- user_upload/
|   |       `-- .gitkeep
|   `-- results/
|       `-- .gitkeep
|-- main.py
|-- install_commands.txt
|-- requirements-local.txt
|-- README.md
|-- .gitignore
`-- LICENSE-CODEFORMER-NOTE.md
```

## Troubleshooting

If you get NumPy compatibility errors, run:

```powershell
python -m pip install numpy==1.26.4 --force-reinstall
```

If PyTorch does not detect CUDA, check your NVIDIA drivers, CUDA-compatible PyTorch install, and GPU availability. You can still use CPU execution, but it will be slower.

If model weights are missing, run the script and let the local setup download them if configured, or download the required CodeFormer weights according to the original project instructions.

## Credits

Original CodeFormer project:

https://github.com/sczhou/CodeFormer

All original CodeFormer code and models remain credited to their original authors.
