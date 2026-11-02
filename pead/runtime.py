"""Validate the supported, pinned ML runtime before lengthy dataset processing."""
import argparse
from importlib import metadata
import sys

INSTALL_COMMAND = 'powershell -ExecutionPolicy Bypass -File scripts/install-windows.ps1'
PINNED_ML = {'anomalib': '2.0.0', 'torch': '2.5.1', 'torchvision': '0.20.1'}


def validate_runtime(require_ml=True, version_info=None):
    version = tuple((version_info or sys.version_info)[:2])
    if not (3, 10) <= version < (3, 13):
        raise RuntimeError(
            f'Unsupported Python {version[0]}.{version[1]}. PEAD requires Python 3.10–3.12 '
            f'(Windows installer uses 3.11). Install Python 3.11 x64, run `{INSTALL_COMMAND}`, '
            'then use scripts/start.ps1. Do not launch with the global Python 3.13 interpreter.'
        )
    versions = {}
    if require_ml:
        for package, expected in PINNED_ML.items():
            try:
                actual = metadata.version(package)
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f'Missing {package}. Run `{INSTALL_COMMAND}` and start with scripts/start.ps1 '
                    'to use the project .venv.'
                ) from exc
            if actual.split('+', 1)[0] != expected:
                raise RuntimeError(
                    f'Unsupported {package} {actual}; this application requires {package}=={expected}. '
                    f'Run `{INSTALL_COMMAND}` and start with scripts/start.ps1 to use the project .venv.'
                )
            versions[package] = actual
    return {'python': f'{version[0]}.{version[1]}', 'executable': sys.executable, 'versions': versions}


def symlink_free_engine_class():
    """Keep anomalib's lifecycle, but use ordinary folders inside unique run roots.

    anomalib 2.0's Engine.fit always requests a versioned workspace with a
    `latest` symlink. The application already assigns each job a unique run
    directory, so another versioning layer is unnecessary. Its non-versioned
    workspace branch uses an ordinary `latest` directory and needs no Windows
    symlink privilege. Pinning anomalib protects this narrow API override.
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
        print(f"Runtime verified: Python {info['python']} — {info['executable']}")
        print('CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
        else:
            print('CPU fallback active; check the NVIDIA driver if GPU acceleration is expected.')
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == '__main__':
    main()
