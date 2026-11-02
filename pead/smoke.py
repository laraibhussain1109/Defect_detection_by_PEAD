"""Real model integration smoke, deliberately using synthetic test-only data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from .imaging import Recipe, overlay
from .train import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',required=True)
    parser.add_argument('--verify-existing',action='store_true')
    parser.add_argument('--image-size',type=int,choices=(256,512),default=256)
    parser.add_argument('--train-tile-budget',type=int,default=512)
    parser.add_argument('--validation-tile-budget',type=int,default=128)
    parser.add_argument('--quick-check',action='store_true')
    args=parser.parse_args()
    base=Path(args.directory).resolve()
    if args.verify_existing:
        verify(base)
        return
    if base.exists():
        raise SystemExit('Use a new disposable directory; existing files are never overwritten')
    good=base/'good'
    good.mkdir(parents=True)
    rng=np.random.default_rng(42)
    for i in range(30):
        image=np.clip(rng.normal(115,3,(256,256,3)),0,240).astype(np.uint8)
        cv2.circle(image,(128,128),65,(145+i%4,)*3,-1)
        cv2.line(image,(70,65),(185,65),(85,85,85),3)
        if args.image_size != 256:
            image=cv2.resize(image,(args.image_size,args.image_size))
        cv2.imwrite(str(good/f'{i}.png'),image)
    # Test fixture only: sufficient for exercising the penalty-image dataloader,
    # explicitly not a substitute for the official ImageNette training dataset.
    penalty=base/'models'/'assets'/'imagenette'/'fixture'
    penalty.mkdir(parents=True)
    for i in range(8):
        image=rng.integers(0,240,(512,512,3),dtype=np.uint8)
        cv2.imwrite(str(penalty/f'{i}.png'),image)
    run=base/'models'/'smoke'
    run.mkdir()
    job={'dataset':str(good),'run':str(run),'recipe':Recipe(tile_size=256).dict(),'epochs':1,'coreset':.001,
         'train_tile_budget':args.train_tile_budget,'validation_tile_budget':args.validation_tile_budget,
         'quick_check':args.quick_check}
    atomic_json(run/'job.json',job)
    subprocess.run([sys.executable,'-m','pead.train',str(run/'job.json')],check=True)
    verify(base)


def verify(base):
    active=json.loads((base/'models'/'active.json').read_text())
    run=base/'models'/active['run']
    good=base/'good'
    from .model import Hybrid
    hybrid=Hybrid(run,device='cpu')
    frame=cv2.imread(str(good/'0.png'))
    assert all(raw.shape==(256,256) for raw in hybrid.tile_maps(frame)), 'Both models must cover the full tile'
    normal,_,_=hybrid.predict(frame)
    cv2.line(frame,(80,110),(175,160),(10,10,10),5)
    score,glare,box=hybrid.predict(frame)
    assert np.isfinite(score).all() and score.shape==frame.shape[:2]
    assert not np.allclose(normal,score), 'Changed pixels must affect predictions'
    image,result=overlay(frame,score,glare,box,hybrid.recipe)
    cv2.imwrite(str(base/'smoke-overlay.jpg'),image)
    # Exercise the actual HTTP lifecycle and asynchronous inference loop using
    # a simulated camera source. Physical USB access remains a separate test.
    os.environ['PEAD_DATA_DIR']=str(base)
    from fastapi.testclient import TestClient
    from . import app as server
    import time
    server.camera.snapshot=lambda:(frame.copy(),int(time.monotonic()*1000))
    with TestClient(server.app) as client:
        response=client.post('/api/inspection/start')
        assert response.status_code==200,response.text
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            status=client.get('/api/status').json()
            if status['result']:
                break
            assert not status['error'],status['error']
            time.sleep(.2)
        assert status['result'], 'Live inference did not produce a result'
        assert status['model_quick_check']==hybrid.meta.get('quick_check',False)
        assert client.get('/api/frame').headers['content-type']=='image/jpeg'
        assert client.post('/api/inspection/stop').status_code==200
        assert client.get('/api/status').json()['result'] is None
    print(json.dumps({'smoke':'passed','models':hybrid.meta['models'],'synthetic_decision':result['decision'],
                      'note':'Integration check only; no real defect accuracy claim.'},indent=2))

if __name__=='__main__':
    main()
