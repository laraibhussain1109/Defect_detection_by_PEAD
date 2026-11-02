from contextlib import asynccontextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .camera import Camera
from .imaging import Recipe, overlay, prepare, preview_mask
from .dataset import (DEFAULT_TRAIN_TILES, DEFAULT_VALIDATION_TILES, MAX_TRAIN_TILES,
                      MAX_VALIDATION_TILES, EXTENSIONS, training_settings)
from .train import atomic_json

DATA = Path(os.environ.get('PEAD_DATA_DIR','data')).resolve()
DATA.mkdir(parents=True,exist_ok=True)
(DATA/'captures').mkdir(exist_ok=True)
(DATA/'models').mkdir(exist_ok=True)
STATIC = Path(__file__).parent/'static'
camera = Camera()
state_lock = threading.RLock()
state = {'inspection':False,'processing':False,'result':None,'jpeg':None,'error':None,'model':None,'run':None,'inference_ms':None}
process = None
log_handle = None
current_run = None
shutdown = threading.Event()


def training_active():
    return process is not None and process.poll() is None


def inspect_loop():
    previous = -1
    while not shutdown.wait(.01):
        with state_lock:
            enabled = state['inspection'] and not training_active()
            model = state['model']
        if not enabled or model is None:
            continue
        frame,seq = camera.snapshot()
        if frame is None or seq==previous:
            continue
        previous = seq
        start = time.monotonic()
        captured_at = datetime.now(timezone.utc).isoformat()
        with state_lock:
            if not state['inspection']:
                continue
            state['processing'] = True
        try:
            score,glare,box = model.predict(frame,cancel=lambda:shutdown.is_set() or not state['inspection'])
            image,result = overlay(frame,score,glare,box,model.recipe)
            elapsed = (time.monotonic()-start)*1000
            # Drop output from a previous camera/model/inspection session.
            with state_lock:
                if not state['inspection'] or state['model'] is not model:
                    continue
                result.update({'timestamp':datetime.now(timezone.utc).isoformat(),'captured_at':captured_at,
                               'inference_ms':elapsed,'sequence':seq,'model':state['run']})
                state.update(result=result,jpeg=encode(image),error=None,inference_ms=elapsed)
            with (DATA/'inspection.jsonl').open('a') as f:
                f.write(json.dumps(result)+'\n')
        except InterruptedError:
            pass
        except Exception as exc:
            with state_lock:
                state.update(error=str(exc),inspection=False,result=None,jpeg=None)
        finally:
            with state_lock:
                state['processing'] = False


def encode(image):
    # Preview is downsized; inference and saved captures retain camera resolution.
    if image.shape[1]>1440:
        image = cv2.resize(image,(1440,round(image.shape[0]*1440/image.shape[1])))
    ok,jpeg = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return jpeg.tobytes()


@asynccontextmanager
async def lifespan(app):
    shutdown.clear()
    thread = threading.Thread(target=inspect_loop,daemon=True)
    thread.start()
    yield
    shutdown.set()
    if training_active():
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
    try:
        camera.stop()
    except RuntimeError:
        pass
    # A stop request interrupts inference between tiles, instead of leaving a
    # daemon thread in PyTorch while Python tears down the process.
    thread.join(timeout=30)
    if log_handle:
        log_handle.close()


app = FastAPI(title='PEAD Inspection Workstation',lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])


@app.middleware('http')
async def local_only(request: Request,call_next):
    # Prevent a remote website from issuing camera/training commands to localhost.
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        return Response('Cross-origin access denied',status_code=403)
    if request.headers.get('sec-fetch-site')=='cross-site':
        return Response('Cross-site access denied',status_code=403)
    return await call_next(request)


app.mount('/static',StaticFiles(directory=STATIC),name='static')


@app.get('/')
def index():
    return FileResponse(STATIC/'index.html')


class CameraConfig(BaseModel):
    index: int = Field(default=0,ge=0,le=16)
    width: int = Field(default=3840,ge=320,le=4096)
    height: int = Field(default=2160,ge=240,le=2160)
    fps: int = Field(default=30,ge=1,le=60)


@app.post('/api/camera/start')
def start_camera(config: CameraConfig):
    with state_lock:
        state.update(inspection=False,jpeg=None,result=None)
        try:
            camera.start(**config.model_dump())
        except RuntimeError as exc:
            raise HTTPException(409,str(exc)) from exc
    return {'state':'connecting'}


@app.post('/api/camera/stop')
def stop_camera():
    with state_lock:
        state.update(inspection=False,jpeg=None,result=None)
        try:
            camera.stop()
        except RuntimeError as exc:
            raise HTTPException(409,str(exc)) from exc
    return {'state':'disconnected'}


@app.get('/api/status')
def status():
    frame,_ = camera.snapshot()
    with state_lock:
        result = state['result'] if frame is not None and state['inspection'] else None
        if result:
            result = dict(result)
            age = (datetime.now(timezone.utc)-datetime.fromisoformat(result['captured_at'])).total_seconds()
            result['frame_age_ms'] = age*1000
            if age>2:
                result['decision'] = 'STALE — WAIT FOR INFERENCE'
        active = DATA/'models'/'active.json'
        available_run = json.loads(active.read_text())['run'] if active.exists() else None
        if state['model'] is not None:
            model_quick_check = state['model'].meta.get('quick_check',False)
        elif available_run and (DATA/'models'/available_run/'manifest.json').exists():
            model_quick_check = json.loads((DATA/'models'/available_run/'manifest.json').read_text()).get('quick_check',False)
        else:
            model_quick_check = False
        return {'camera':{'connected':frame is not None,'error':camera.error,'actual':camera.actual},
                'inspection':state['inspection'],'result':result,'error':state['error'],
                'inference_ms':state['inference_ms'],'model':state['run'],
                'available_model':available_run,'model_quick_check':model_quick_check,
                'ml_installed':importlib.util.find_spec('anomalib') is not None,
                'captures':len(list((DATA/'captures').glob('*.png'))),'capture_folder':str(DATA/'captures'),
                'training':job_status()}


def job_status():
    if current_run is None:
        return {'state':'idle'}
    quality_path = current_run/'quality.json'
    extra = {}
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding='utf-8'))
        extra['quality'] = {key:quality[key] for key in ('scanned_files','distinct_sources',
            'accepted_sources','rejected_sources','duplicate_files')}
    result = current_run/'result.json'
    if training_active():
        return {'state':'running','run':current_run.name,**extra}
    if result.exists():
        return {**json.loads(result.read_text()),'run':current_run.name,**extra}
    return {'state':'failed','error':'Training process exited; inspect the log','run':current_run.name,**extra}


@app.get('/api/frame')
def frame_preview():
    frame,_ = camera.snapshot()
    if frame is None:
        raise HTTPException(503,'No fresh camera frame')
    with state_lock:
        jpeg = state['jpeg'] if state['inspection'] else None
    return Response(jpeg or encode(frame),media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@app.post('/api/capture')
def capture():
    frame,_ = camera.snapshot()
    if frame is None:
        raise HTTPException(409,'Connect camera first')
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.png'
    if not cv2.imwrite(str(DATA/'captures'/name),frame):
        raise HTTPException(500,'Could not save capture')
    return {'file':name}


@app.post('/api/dataset/upload')
async def upload(file: UploadFile = File(...)):
    content = await file.read(20*1024*1024+1)
    await file.close()
    if len(content)>20*1024*1024:
        raise HTTPException(413,'Each image must be under 20 MB')
    # Pillow checks dimensions before OpenCV allocates the full decoded image.
    from PIL import Image
    from io import BytesIO
    try:
        with Image.open(BytesIO(content)) as image:
            if image.width*image.height>25_000_000:
                raise ValueError('Image exceeds 25 megapixels')
            image.verify()
        image = cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Unsupported image')
    except Exception as exc:
        raise HTTPException(400,'Invalid image: '+str(exc)) from exc
    name = uuid.uuid4().hex+'.png'
    if not cv2.imwrite(str(DATA/'captures'/name),image):
        raise HTTPException(500,'Could not save uploaded image')
    return {'file':name}


class TrainingConfig(BaseModel):
    dataset: str = ''
    epochs: int = Field(default=100,ge=1,le=1000)
    coreset: float = Field(default=.01,ge=.001,le=.1)
    train_tile_budget: int = Field(default=DEFAULT_TRAIN_TILES,ge=24,le=MAX_TRAIN_TILES)
    validation_tile_budget: int = Field(default=DEFAULT_VALIDATION_TILES,ge=6,le=MAX_VALIDATION_TILES)
    quick_check: bool = False
    roi: tuple[float,float,float,float] = (0,0,1,1)
    clahe: bool = False
    glare_level: int = 248
    max_glare: float = .02
    tile_size: int = 256
    min_area: int = 4
    green_background: bool = True
    green_hue_low: int = 35
    green_hue_high: int = 95
    green_saturation_min: int = 60
    green_value_min: int = 35
    min_part_fraction: float = .02


def recipe_from_config(config):
    return Recipe(**config.model_dump(exclude={'dataset','epochs','coreset','train_tile_budget',
                                              'validation_tile_budget','quick_check'}))


@app.post('/api/recipe/preview')
def recipe_preview(config: TrainingConfig):
    try:
        recipe = recipe_from_config(config)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    frame,_ = camera.snapshot()
    if frame is None:
        dataset = Path(config.dataset).expanduser().resolve() if config.dataset else DATA/'captures'
        if not dataset.is_dir():
            raise HTTPException(422,'Dataset folder does not exist on this computer')
        source = next((p for p in sorted(dataset.rglob('*')) if p.is_file() and p.suffix.lower() in EXTENSIONS),None)
        if source is None:
            raise HTTPException(409,'Connect the camera or select a folder containing an image to preview the part mask')
        from PIL import Image
        try:
            with Image.open(source) as image:
                if image.width*image.height>25_000_000:
                    raise ValueError('Image exceeds 25 megapixels')
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError('Cannot decode the preview image')
        except Exception as exc:
            raise HTTPException(422,str(exc)) from exc
    try:
        preview,fraction = preview_mask(frame,recipe)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    return Response(encode(preview),media_type='image/jpeg',
                    headers={'Cache-Control':'no-store','X-Part-Fraction':str(fraction)})


@app.post('/api/train')
def start_training(config: TrainingConfig):
    global process,log_handle,current_run
    with state_lock:
        if training_active() or state['inspection'] or state['processing']:
            raise HTTPException(409,'Stop inspection and wait for the current frame to finish, or wait for current training')
        if importlib.util.find_spec('anomalib') is None:
            raise HTTPException(409,'ML dependencies missing. Follow the installation instructions in README.md.')
        try:
            recipe = recipe_from_config(config)
            settings = training_settings(config.model_dump())
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        dataset = Path(config.dataset).expanduser().resolve() if config.dataset else DATA/'captures'
        if not dataset.is_dir():
            raise HTTPException(422,'Dataset folder does not exist on this computer')
        files = [p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}]
        if not 30<=len(files)<=2000:
            raise HTTPException(422,'Select 30–2000 good images; duplicates are rejected during training')
        current_run = DATA/'models'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
        current_run.mkdir()
        job = {'dataset':str(dataset),'run':str(current_run),'recipe':recipe.dict(),**settings}
        atomic_json(current_run/'job.json',job)
        if log_handle:
            log_handle.close()
        log_handle = (current_run/'train.log').open('w')
        # Release prior model/GPU memory before starting training.
        if state['model'] is not None:
            for model in state['model'].models:
                model.cpu()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state.update(model=None,run=None,jpeg=None,result=None,error=None)
        process = subprocess.Popen([sys.executable,'-u','-m','pead.train',str(current_run/'job.json')],stdout=log_handle,stderr=subprocess.STDOUT)
    return {'state':'running','run':current_run.name,**settings}


@app.get('/api/train/log')
def training_log():
    if current_run is None:
        return {'text':'No training started.'}
    path = current_run/'train.log'
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-16000))
        return {'text':f.read().decode(errors='replace')}


@app.post('/api/train/cancel')
def cancel_training():
    with state_lock:
        if training_active():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            atomic_json(current_run/'result.json',{'state':'cancelled'})
    return {'state':'cancelled'}


@app.post('/api/inspection/start')
def start_inspection():
    with state_lock:
        if training_active():
            raise HTTPException(409,'Training is in progress')
        if camera.snapshot()[0] is None:
            raise HTTPException(409,'Connect the camera first')
        if state['inspection']:
            return {'state':'inspecting'}
        if state['processing']:
            raise HTTPException(409,'Wait for the previous inspection frame to finish')
        active = DATA/'models'/'active.json'
        if not active.exists():
            raise HTTPException(409,'Train a model first')
        try:
            from .model import Hybrid
            run = json.loads(active.read_text())['run']
            model = state['model'] if state['run']==run and state['model'] is not None else Hybrid(DATA/'models'/run)
        except Exception as exc:
            raise HTTPException(500,'Model load failed: '+str(exc)) from exc
        state.update(model=model,run=run,inspection=True,error=None,result=None,jpeg=None)
    return {'state':'inspecting','recipe':model.recipe.dict()}


@app.post('/api/inspection/stop')
def stop_inspection():
    with state_lock:
        state.update(inspection=False,result=None,jpeg=None)
    return {'state':'stopped'}
