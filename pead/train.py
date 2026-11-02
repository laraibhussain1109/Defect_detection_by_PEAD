"""Isolated training process; only publishes a complete model pair."""
import argparse
import json
import os
from pathlib import Path
import traceback
import cv2
import numpy as np
from .imaging import Recipe
from .dataset import stage_dataset, training_settings
from .runtime import validate_runtime, symlink_free_engine_class


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2))
    os.replace(temp,path)


def model_assets_dir(run_root):
    """Windows launchers supply a short cache; other workflows keep their cache."""
    return Path(os.environ.get('PEAD_ASSETS_DIR', str(Path(run_root).parent/'assets'))).expanduser().resolve()


def calibration_values(raw, mask_path, required=False):
    """Select only inspectable part pixels from a staged held-out tile."""
    mask_path = Path(mask_path)
    if mask_path.exists():
        part_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if part_mask is None:
            raise ValueError(f'Unreadable calibration part mask: {mask_path.name}')
        valid = cv2.resize(part_mask, raw.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        return raw[valid]
    if required:
        raise ValueError(f'Missing calibration part mask: {mask_path.name}; rerun training')
    return raw.ravel()


def train(job):
    runtime = validate_runtime()
    print(f"Training runtime: Python {runtime['python']} — {runtime['executable']}", flush=True)
    root = Path(job['run']).resolve()
    recipe = Recipe(**job['recipe'])
    settings = training_settings(job)
    if settings['quick_check']:
        print('QUICK CHECK: 1 epoch with bounded representative tiles. '
              'This checks training/inference plumbing, not defect accuracy.', flush=True)
    sampling = stage_dataset(root, Path(job['dataset']).resolve(), recipe,
                             settings['train_tile_budget'], settings['validation_tile_budget'])
    atomic_json(root/'sampling.json', sampling)
    source_shape = sampling['source_shape']
    source_splits = sampling['splits']
    import torch
    from lightning import seed_everything
    from lightning.pytorch.callbacks import Callback
    from anomalib.data import Folder
    from anomalib.models import EfficientAd, Patchcore
    Engine = symlink_free_engine_class()
    seed_everything(42,workers=True)
    torch.set_num_threads(min(8,os.cpu_count() or 1))

    class Progress(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            print(f'{pl_module.__class__.__name__}: epoch {trainer.current_epoch+1}/{trainer.max_epochs}', flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            completed = batch_idx+1
            if completed % 25 == 0 or completed == trainer.num_training_batches:
                print(f'{pl_module.__class__.__name__}: epoch {trainer.current_epoch+1}, '
                      f'tile {completed}/{trainer.num_training_batches}', flush=True)

    assets = model_assets_dir(root)
    assets.mkdir(parents=True, exist_ok=True)
    print(f'Model assets: {assets}',flush=True)
    # anomalib stores its teacher cache relative to cwd; keep it out of source.
    os.chdir(assets)
    torch.hub.set_dir(str(assets/'torch'))
    models = [EfficientAd(imagenet_dir=str(assets/'imagenette'),post_processor=False,evaluator=False,visualizer=False),
              Patchcore(backbone='resnet18',layers=['layer2','layer3'],pre_trained=False,coreset_sampling_ratio=settings['coreset'],
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
        engine = Engine(default_root_dir=root/name.replace('.ckpt',''),max_epochs=1 if name=='patchcore.ckpt' else settings['epochs'],
                        accelerator='auto',devices=1,logger=False,enable_progress_bar=False,callbacks=[Progress()])
        engine.fit(model=model,datamodule=dm)
        engine.trainer.save_checkpoint(root/name)
        model.cpu().eval()
        # The modern PatchCore model owns its transient embedding store. Keep
        # those plain-list tensors from retaining accelerator memory.
        if isinstance(model, Patchcore):
            model.model.embedding_store.clear()
        del engine, dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Calibrate each raw model map on untouched good source images, separate from training.
    calibration = []
    print('Calibrating both models on held-out source tiles', flush=True)
    for model in models:
        samples = []
        with torch.inference_mode():
            for p in sorted((root/'dataset'/'validation').glob('*.png')):
                image = cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(image.copy()).permute(2,0,1).float().unsqueeze(0)/255
                raw = model(tensor).anomaly_map.squeeze().numpy()
                mask_path = root/'dataset_masks'/'validation'/p.name
                values = calibration_values(raw, mask_path, required=recipe.green_background)
                if values.size:
                    samples.append(values[::max(1,values.size//4096)])
        if not samples:
            raise ValueError('No visible part pixels remain for model calibration')
        values = np.concatenate(samples)
        median,upper = np.quantile(values,[.5,.9995])
        if not np.isfinite(values).all() or upper-median<1e-9:
            raise ValueError('Degenerate calibration: model maps do not distinguish normal variation')
        calibration.append({'median':float(median),'upper':float(upper)})
    manifest = {'format':2,'runtime':runtime,'recipe':recipe.dict(),'source_shape':source_shape,'calibration':calibration,'normal_sources':source_splits['normal']['available_sources'],
                'validation_sources':source_splits['validation']['available_sources'],'epochs':settings['epochs'],
                'quick_check':settings['quick_check'],
                'quality':sampling['quality'],
                'tile_sampling':{name:{key:value for key,value in split.items() if key!='sources'}
                                 for name,split in source_splits.items()},'models':['EfficientAd-small','Patchcore-resnet18'],
                'threshold_note':'Normal-only pixel quantiles; validate with real defects before production acceptance.'}
    atomic_json(root/'manifest.json',manifest)
    # Publish only after checkpoints can be loaded and a real prediction succeeds.
    from .model import Hybrid
    hybrid = Hybrid(root,device='cpu')
    sample_path = next(root.glob('dataset/normal/*.png'))
    sample_frame = cv2.imread(str(sample_path))
    maps = hybrid.tile_maps(sample_frame)
    score = np.maximum(maps[0], maps[1])
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
