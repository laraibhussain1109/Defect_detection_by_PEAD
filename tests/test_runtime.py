from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pead import runtime
from pead.train import calibration_values


def test_python313_is_rejected_before_dataset_staging(monkeypatch):
    from pead import train
    monkeypatch.setattr(runtime.sys, 'version_info', (3,13,0))
    monkeypatch.setattr(train, 'stage_dataset', lambda *args, **kwargs: pytest.fail('Dataset staging must not begin'))
    with pytest.raises(RuntimeError, match='Unsupported Python 3.13'):
        train.train({})


def test_wrong_core_dependency_has_actionable_repair_message(monkeypatch):
    versions = {**runtime.PINNED_ML, 'torch':'2.10.0'}
    monkeypatch.setattr(runtime.metadata, 'version', lambda package: versions[package])
    with pytest.raises(RuntimeError, match=r'torch==2.5.1.*install-windows.ps1'):
        runtime.validate_runtime(version_info=(3,11,0))


def test_supported_cpu_build_is_accepted(monkeypatch):
    versions = {**runtime.PINNED_ML, 'torch':'2.5.1+cpu', 'torchvision':'0.20.1+cpu'}
    monkeypatch.setattr(runtime.metadata, 'version', lambda package: versions[package])
    result = runtime.validate_runtime(version_info=(3,12,0))
    assert result['versions']['torch']=='2.5.1+cpu'


def test_workspace_works_when_symlinks_are_denied(tmp_path,monkeypatch):
    # Exercise the actual anomalib workspace code that raised WinError1314.
    # The before check demonstrates that the fixture reproduces that failure.
    from anomalib.engine import Engine
    def denied(*args, **kwargs):
        raise OSError(1314, 'A required privilege is not held by the client')
    monkeypatch.setattr(Path, 'symlink_to', denied)
    model = SimpleNamespace(name='EfficientAd')
    data = SimpleNamespace(name='part', category='')
    before = Engine(default_root_dir=tmp_path/'before', logger=False)
    with pytest.raises(OSError, match='privilege'):
        before._setup_workspace(model=model, datamodule=data, versioned_dir=True)
    engine = runtime.symlink_free_engine_class()(default_root_dir=tmp_path/'fixed', logger=False)
    engine._setup_workspace(model=model, datamodule=data, versioned_dir=True)
    workspace = engine._cache.args['default_root_dir']
    workspace.mkdir(parents=True)
    assert workspace==tmp_path/'fixed'/'EfficientAd'/'part'/'latest'
    assert workspace.is_dir() and not workspace.is_symlink()
    assert not list((tmp_path/'fixed').rglob('v0'))


def test_background_outliers_are_excluded_from_calibration(tmp_path):
    raw = np.full((4,4), 10000, np.float32)
    raw[1:3,1:3] = 2
    mask = np.zeros((8,8),np.uint8)
    mask[2:6,2:6] = 255
    path = tmp_path/'mask.png'
    assert cv2.imwrite(str(path),mask)
    values = calibration_values(raw,path,required=True)
    assert values.shape==(4,)
    assert np.all(values==2)
    with pytest.raises(ValueError,match='Missing calibration part mask'):
        calibration_values(raw,tmp_path/'missing.png',required=True)
