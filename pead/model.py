"""Actual anomalib models. Imported only when ML is requested."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .imaging import Recipe, prepare, tiles


class Hybrid:
    def __init__(self, directory, device=None):
        import torch
        from anomalib.models import EfficientAd, Patchcore
        self.torch = torch
        self.directory = Path(directory)
        self.meta = json.loads((self.directory/'manifest.json').read_text())
        self.recipe = Recipe(**self.meta['recipe'])
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device=='cpu':
            torch.set_num_threads(min(4,os.cpu_count() or 1))
        # Checkpoints are locally generated trusted artifacts, never user uploads.
        self.models = [EfficientAd.load_from_checkpoint(self.directory/'efficientad.ckpt',map_location='cpu'),
                       Patchcore.load_from_checkpoint(self.directory/'patchcore.ckpt',map_location='cpu',pre_trained=False)]
        for model in self.models:
            model.to(self.device).eval()

    def tile_maps(self, tile):
        rgb = cv2.cvtColor(tile,cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb.copy()).permute(2,0,1).float().unsqueeze(0).to(self.device)/255
        with self.torch.inference_mode():
            return [m(tensor).anomaly_map.squeeze().cpu().numpy() for m in self.models]

    def predict(self, frame, cancel=None):
        if list(frame.shape[:2]) != self.meta['source_shape']:
            raise ValueError('Camera resolution differs from training; restore the trained capture profile or retrain')
        image,glare,box = prepare(frame,self.recipe)
        fused = np.zeros(image.shape[:2],np.float32)
        for (x,y,x1,y1),tile in tiles(image,self.recipe):
            if cancel is not None and cancel():
                raise InterruptedError('Inspection stopped')
            maps = self.tile_maps(tile)
            if cancel is not None and cancel():
                raise InterruptedError('Inspection stopped')
            if any(not np.isfinite(raw).all() for raw in maps):
                raise ValueError('Model produced non-finite anomaly scores; inspection stopped')
            normalized = []
            for raw,cal in zip(maps,self.meta['calibration']):
                scaled = np.maximum(0,(raw-cal['median'])/(cal['upper']-cal['median']))
                normalized.append(cv2.resize(scaled,(x1-x,y1-y)))
            # Conservative OR fusion. A defect detected by either model survives.
            score = np.maximum(*normalized)
            fused[y:y1,x:x1] = np.maximum(fused[y:y1,x:x1],score)
        return fused,glare,box
