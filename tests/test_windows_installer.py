"""Execute installer/launcher logic under PowerShell with isolated native-command fixtures."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_installer_and_launcher_use_short_paths_and_block_partial_installs(tmp_path):
    # This harness uses POSIX executable fixtures for Python/pip. It exercises
    # real PowerShell scripts, not Windows NTFS or a physical CUDA installation.
    pwsh = os.environ.get('PEAD_PWSH') or shutil.which('pwsh')
    if os.name != 'posix' or not pwsh:
        pytest.skip('Requires PowerShell and POSIX native-command fixtures')
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ('PEAD_RUNTIME_ROOT', 'PEAD_ASSETS_DIR'):
        env.pop(key, None)
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME'):
        env[key] = str(tmp_path / key.lower())
    result = subprocess.run([pwsh, '-NoProfile', '-File', str(root/'tests/windows-installer.Tests.ps1'),
                             '-ProjectRoot', str(root)],env=env,text=True,capture_output=True,timeout=60)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert 'PowerShell installer/launcher assertions' in result.stdout
