"""Isolated training process; only publishes a complete model pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import traceback
import cv2
import numpy as np
from .imaging import Recipe, prepare, tiles


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2))
    os.replace(temp,path)


def train(job):
    import torch
    from lightning import seed_everything
    from anomalib.data import Folder
    from anomalib.models import EfficientAd, Patchcore
    from anomalib.engine import Engine
    seed_everything(42,workers=True)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    root = Path(job['run'])
    recipe = Recipe(**job['recipe'])
    dataset = Path(job['dataset']).resolve()
    files = sorted(p for p in dataset.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'} and p.is_file())
    if len(files)>2000:
        raise ValueError('Maximum 2000 source images per run; curate a representative dataset first')
    # Deduplicate before splitting: repeated camera captures must not leak to validation.
    unique = {}
    for p in files:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        unique.setdefault(digest,p)
    files = list(unique.values())
    if len(files)<30:
        raise ValueError('At least 30 distinct good images are required (100–500 recommended)')
    random.Random(42).shuffle(files)
    nval = max(6,len(files)//5)
    splits = {'validation':files[:nval], 'normal':files[nval:]}
    source_shape = None
    for name,paths in splits.items():
        target = root/'dataset'/name
        target.mkdir(parents=True)
        total = 0
        for i,p in enumerate(paths):
            frame = cv2.imread(str(p))
            if frame is None:
                raise ValueError(f'Unreadable image: {p.name}')
            if source_shape is None:
                source_shape = list(frame.shape[:2])
            if list(frame.shape[:2]) != source_shape:
                raise ValueError('All source images must have the same camera resolution')
            image,glare,_ = prepare(frame,recipe)
            if glare.mean()>recipe.max_glare:
                raise ValueError(f'Excess glare in {p.name}; recapture under controlled lighting')
            for j,(_,tile) in enumerate(tiles(image,recipe)):
                total += 1
                if total>4000:
                    raise ValueError('More than 4000 tiles per split; reduce dataset or narrow ROI')
                if not cv2.imwrite(str(target/f'{i:05}_{j:03}.png'),tile):
                    raise OSError('Could not write staged training image')
    assets = root.parent/'assets'
    assets.mkdir(exist_ok=True)
    # anomalib stores its teacher cache relative to cwd; keep it out of source.
    os.chdir(assets)
    torch.hub.set_dir(str(assets/'torch'))
    models = [EfficientAd(imagenet_dir=str(root.parent/'assets'/'imagenette'),post_processor=False,evaluator=False,visualizer=False),
              Patchcore(backbone='resnet18',layers=['layer2','layer3'],pre_trained=False,coreset_sampling_ratio=job['coreset'],
                        pre_processor=Patchcore.configure_pre_processor(image_size=(256,256),center_crop_size=(256,256)),
                        post_processor=False,evaluator=False,visualizer=False)]
    # Use official, hash-checked torchvision ResNet18 weights with the equivalent
    # timm feature architecture; this also makes checkpoint reload fully offline.
    from torchvision.models import ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True,check_hash=True)
    extractor = models[1].model.feature_extractor.feature_extractor
    # Older official checkpoints omit BatchNorm's training-step counters;
    # PyTorch supplies those zero counters during strict loading.
    extractor.load_state_dict({key:weights[key] for key in extractor.state_dict() if key in weights},strict=True)
    checkpoints = ['efficientad.ckpt','patchcore.ckpt']
    for model,name in zip(models,checkpoints):
        print(f'\nTraining {name}',flush=True)
        dm = Folder(name='part',root=root/'dataset',normal_dir='normal',normal_test_dir='validation',
                    normal_split_ratio=0,train_batch_size=1,eval_batch_size=1,num_workers=0,
                    val_split_mode='same_as_test',seed=42)
        engine = Engine(default_root_dir=root/name.replace('.ckpt',''),max_epochs=1 if name=='patchcore.ckpt' else job['epochs'],
                        accelerator='auto',devices=1,logger=False,enable_progress_bar=False)
        engine.fit(model=model,datamodule=dm)
        engine.trainer.save_checkpoint(root/name)
        model.cpu().eval()
    # Calibrate each raw model map on untouched good source images, separate from training.
    calibration = []
    for model in models:
        samples = []
        with torch.inference_mode():
            for p in sorted((root/'dataset'/'validation').glob('*.png')):
                image = cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(image.copy()).permute(2,0,1).float().unsqueeze(0)/255
                raw = model(tensor).anomaly_map.squeeze().numpy()
                samples.append(raw.ravel()[::max(1,raw.size//4096)])
        values = np.concatenate(samples)
        median,upper = np.quantile(values,[.5,.9995])
        if not np.isfinite(values).all() or upper-median<1e-9:
            raise ValueError('Degenerate calibration: model maps do not distinguish normal variation')
        calibration.append({'median':float(median),'upper':float(upper)})
    manifest = {'format':1,'recipe':recipe.dict(),'source_shape':source_shape,'calibration':calibration,'normal_sources':len(files)-nval,
                'validation_sources':nval,'epochs':job['epochs'],'models':['EfficientAd-small','Patchcore-resnet18'],
                'threshold_note':'Normal-only pixel quantiles; validate with real defects before production acceptance.'}
    atomic_json(root/'manifest.json',manifest)
    # Publish only after checkpoints can be loaded and a real prediction succeeds.
    from .model import Hybrid
    hybrid = Hybrid(root,device='cpu')
    score,_,_ = hybrid.predict(cv2.imread(str(files[0])))
    if not np.isfinite(score).all():
        raise ValueError('Non-finite hybrid prediction')
    atomic_json(root.parent/'active.json',{'run':root.name})
    print('TRAINING COMPLETE: model pair verified and activated',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    try:
        train(job)
        atomic_json(Path(job['run'])/'result.json',{'state':'complete'})
    except Exception as exc:
        traceback.print_exc()
        atomic_json(Path(job['run'])/'result.json',{'state':'failed','error':str(exc)})
        raise SystemExit(1)

if __name__=='__main__':
    main()
