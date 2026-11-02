$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 x64 from python.org first.' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed' }
& .\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& .\.venv\Scripts\python.exe -m pip install -e '.[ml,dev]'
if ($LASTEXITCODE -ne 0) { throw 'Application installation failed' }
& .\.venv\Scripts\python.exe -c "import torch; from anomalib.models import EfficientAd, Patchcore; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback: check NVIDIA driver')"
if ($LASTEXITCODE -ne 0) { throw 'Model import check failed' }
