from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pead import runtime
from pead.train import calibration_values


def test_unsupported_python_is_rejected_before_dataset_staging(monkeypatch):
    from pead import train
    monkeypatch.setattr(runtime.sys, 'version_info', (3,14,0))
    monkeypatch.setattr(train, 'stage_dataset', lambda *args, **kwargs: pytest.fail('Dataset staging must not begin'))
    with pytest.raises(RuntimeError, match='Unsupported Python 3.14'):
        train.train({})


def test_wrong_core_dependency_has_actionable_repair_message(monkeypatch):
    versions = {**runtime.PINNED_ML, 'torch':'2.5.1'}
    monkeypatch.setattr(runtime.metadata, 'version', lambda package: versions[package])
    with pytest.raises(RuntimeError, match=r'torch==2.10.0.*install-windows.ps1'):
        runtime.validate_runtime(version_info=(3,11,0))


def test_supported_cpu_build_is_accepted(monkeypatch):
    versions = {**runtime.PINNED_ML, 'torch':'2.10.0+cpu', 'torchvision':'0.25.0+cpu'}
    monkeypatch.setattr(runtime.metadata, 'version', lambda package: versions[package])
    result = runtime.validate_runtime(version_info=(3,13,0))
    assert result['versions']['torch']=='2.10.0+cpu'


def test_workspace_works_when_symlinks_are_denied(tmp_path,monkeypatch):
    # Each application job already has a unique root; workspace setup must not
    # request any symlink/junction privileges, regardless of upstream fallbacks.
    attempts=[]
    def denied(*args, **kwargs):
        attempts.append(args)
        raise OSError(1314, 'A required privilege is not held by the client')
    monkeypatch.setattr(Path, 'symlink_to', denied)
    model = SimpleNamespace(name='EfficientAd')
    data = SimpleNamespace(name='part', category='')
    engine = runtime.symlink_free_engine_class()(default_root_dir=tmp_path/'fixed', logger=False)
    engine._setup_workspace(model=model, datamodule=data, versioned_dir=True)
    workspace = engine._cache.args['default_root_dir']
    workspace.mkdir(parents=True)
    assert workspace==tmp_path/'fixed'/'EfficientAd'/'part'/'latest'
    assert workspace.is_dir() and not workspace.is_symlink()
    assert not list((tmp_path/'fixed').rglob('v0'))
    assert attempts==[]


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


@pytest.mark.parametrize('version', [(3,11,0), (3,12,0), (3,13,0)])
def test_supported_python_versions_accept_current_stack(monkeypatch, version):
    monkeypatch.setattr(runtime.metadata, 'version', lambda package: runtime.PINNED_ML[package])
    result = runtime.validate_runtime(version_info=version)
    assert result['python'] == f'{version[0]}.{version[1]}'


def test_nvidia_python313_build_is_accepted(monkeypatch):
    versions = {**runtime.PINNED_ML, 'torch':'2.10.0+cu126', 'torchvision':'0.25.0+cu126'}
    monkeypatch.setattr(runtime.metadata, 'version', lambda package: versions[package])
    result = runtime.validate_runtime(version_info=(3,13,0))
    assert result['versions']['torch'] == '2.10.0+cu126'


def test_runtime_diagnostics_are_ascii(monkeypatch):
    with pytest.raises(RuntimeError) as caught:
        runtime.validate_runtime(version_info=(3,14,0))
    assert str(caught.value).isascii()


def test_free_threaded_python_has_actionable_error(monkeypatch):
    monkeypatch.setattr(runtime.sysconfig, 'get_config_var', lambda _:1)
    with pytest.raises(RuntimeError,match='standard Python 3.13'):
        runtime.validate_runtime(require_ml=False,version_info=(3,13,0))


def test_32bit_windows_has_actionable_error(monkeypatch):
    monkeypatch.setattr(runtime.sys, 'platform', 'win32')
    monkeypatch.setattr(runtime.struct, 'calcsize', lambda _:4)
    with pytest.raises(RuntimeError,match='64-bit Python'):
        runtime.validate_runtime(require_ml=False,version_info=(3,13,0))


def test_legacy_models_request_retraining_without_loading_pickle(tmp_path,monkeypatch):
    import json
    from anomalib.models import EfficientAd
    from pead.model import Hybrid
    (tmp_path/'manifest.json').write_text(json.dumps({'format':1}))
    def never(*args,**kwargs):
        pytest.fail('Legacy checkpoint pickle must not be loaded')
    monkeypatch.setattr(EfficientAd,'load_from_checkpoint',never)
    with pytest.raises(ValueError,match='older checkpoint format.*Retrain'):
        Hybrid(tmp_path)
