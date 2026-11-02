import importlib
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setenv('PEAD_DATA_DIR',str(tmp_path))
    import pead.app as module
    module=importlib.reload(module)
    with TestClient(module.app) as client:
        yield client,module


def test_operator_ui_and_missing_camera_are_explicit(app):
    client,module=app
    assert client.get('/').status_code==200
    status=client.get('/api/status').json()
    assert status['camera']['connected'] is False
    assert client.post('/api/capture').status_code==409
    assert client.post('/api/inspection/start').status_code==409
    assert client.get('/api/frame').status_code==503


def test_upload_decodes_image_and_uses_generated_filename(app):
    client,module=app
    image=np.full((100,100,3),80,np.uint8)
    _,png=cv2.imencode('.png',image)
    result=client.post('/api/dataset/upload',files={'file':('../../escape.png',png.tobytes(),'image/png')})
    assert result.status_code==200
    assert '/' not in result.json()['file']
    saved=cv2.imread(str(module.DATA/'captures'/result.json()['file']))
    assert np.array_equal(saved,image)
    assert client.get('/api/status').json()['captures']==1
    assert client.post('/api/dataset/upload',files={'file':('bad.png',b'not an image')}).status_code==400


def test_cross_origin_camera_mutation_is_blocked(app):
    client,_=app
    assert client.post('/api/camera/start',json={},headers={'Origin':'https://evil.example'}).status_code==403


def test_capture_saves_original_resolution_not_preview(app,monkeypatch):
    client,module=app
    frame=np.full((2160,3840,3),100,np.uint8)
    monkeypatch.setattr(module.camera,'snapshot',lambda:(frame,1))
    result=client.post('/api/capture')
    assert result.status_code==200
    saved=cv2.imread(str(module.DATA/'captures'/result.json()['file']))
    assert saved.shape==(2160,3840,3)


def test_stale_capture_is_not_reported_as_current_normal(app,monkeypatch):
    from datetime import datetime,timedelta,timezone
    client,module=app
    monkeypatch.setattr(module.camera,'snapshot',lambda:(np.zeros((64,64,3),np.uint8),1))
    module.state.update(inspection=True,result={
        'decision':'NORMAL','timestamp':datetime.now(timezone.utc).isoformat(),
        'captured_at':(datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat(),
        'regions':[],'glare_fraction':0})
    result=client.get('/api/status').json()['result']
    assert result['decision']=='STALE — WAIT FOR INFERENCE'
    assert result['frame_age_ms']>9000
