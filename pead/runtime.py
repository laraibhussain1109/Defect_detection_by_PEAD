"""Validate the tested Python and ML stack before lengthy dataset processing."""
import argparse
from importlib import metadata
import sys
import struct
import sysconfig

INSTALL_COMMAND = 'powershell -ExecutionPolicy Bypass -File scripts/install-windows.ps1'
PINNED_ML = {'anomalib': '2.7.0', 'torch': '2.10.0', 'torchvision': '0.25.0'}


def validate_runtime(require_ml=True, version_info=None):
    version = tuple((version_info or sys.version_info)[:2])
    if not (3, 11) <= version < (3, 14):
        raise RuntimeError(
            f'Unsupported Python {version[0]}.{version[1]}. PEAD supports Python 3.11-3.13. '
            f'Run `{INSTALL_COMMAND}` to create the project environment. '
            'The installer prefers your existing Python 3.13, then 3.12 or 3.11.'
        )
    if sys.platform == 'win32' and struct.calcsize('P') != 8:
        raise RuntimeError('PEAD requires 64-bit Python on Windows. Install Python 3.13 x64, then rerun the installer.')
    if sysconfig.get_config_var('Py_GIL_DISABLED') in (1, '1'):
        raise RuntimeError('Use standard Python 3.13, not the free-threaded build, for this PyTorch stack. Then rerun the installer.')
    versions = {}
    if require_ml:
        for package, expected in PINNED_ML.items():
            try:
                actual = metadata.version(package)
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f'Missing {package}. Run `{INSTALL_COMMAND}`, then scripts/start.ps1 '
                    'to use the managed PEAD environment.'
                ) from exc
            if actual.split('+', 1)[0] != expected:
                raise RuntimeError(
                    f'Unsupported {package} {actual}; this application requires {package}=={expected}. '
                    f'Run `{INSTALL_COMMAND}`, then scripts/start.ps1 to use the updated managed PEAD environment.'
                )
            versions[package] = actual
    return {'python': f'{version[0]}.{version[1]}', 'executable': sys.executable, 'versions': versions}


def symlink_free_engine_class():
    """Use ordinary workspaces inside already-unique application run folders.

    The pinned anomalib Engine provides a non-versioned workspace branch. Each
    PEAD job already has its own run directory, so no additional symlink or
    junction is needed to select a nested version on Windows.
    """
    from anomalib.engine import Engine

    class RunEngine(Engine):
        def _setup_workspace(self, *args, **kwargs):
            kwargs['versioned_dir'] = False
            return super()._setup_workspace(*args, **kwargs)

    return RunEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check interpreter, dependencies, and model imports')
    parser.parse_args()
    try:
        info = validate_runtime()
        from anomalib.models import EfficientAd, Patchcore
        import torch
        print(f"Runtime verified: Python {info['python']} - {info['executable']}")
        print('CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
        else:
            print('CPU fallback active; check the NVIDIA driver if GPU acceleration is expected.')
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == '__main__':
    main()
