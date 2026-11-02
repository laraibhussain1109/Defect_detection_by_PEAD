"""Bounded, source-separated staging for high-resolution normal-image datasets."""
import hashlib
from pathlib import Path
import random

import cv2
from PIL import Image

from .imaging import prepare, roi_box, tile_boxes

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
DEFAULT_TRAIN_TILES = 512
DEFAULT_VALIDATION_TILES = 128
MAX_TRAIN_TILES = 1024
MAX_VALIDATION_TILES = 512


def training_settings(job):
    """Apply bounded settings to API jobs and legacy on-disk jobs alike."""
    train_budget = int(job.get('train_tile_budget', DEFAULT_TRAIN_TILES))
    val_budget = int(job.get('validation_tile_budget', DEFAULT_VALIDATION_TILES))
    if not 24 <= train_budget <= MAX_TRAIN_TILES:
        raise ValueError(f'Training tile budget must be between 24 and {MAX_TRAIN_TILES}')
    if not 6 <= val_budget <= MAX_VALIDATION_TILES:
        raise ValueError(f'Validation tile budget must be between 6 and {MAX_VALIDATION_TILES}')
    quick = bool(job.get('quick_check', False))
    return {
        'train_tile_budget': min(train_budget, 128) if quick else train_budget,
        'validation_tile_budget': min(val_budget, 64) if quick else val_budget,
        'epochs': 1 if quick else int(job['epochs']),
        'coreset': .001 if quick else float(job['coreset']),
        'quick_check': quick,
    }


def select_tiles(source_count, tiles_per_source, budget, seed=42):
    """Even quotas per source, with deterministic random source/position selection.

    If the budget is smaller than the source count, some sources are unused.
    Otherwise every source is represented, with quotas differing by at most one.
    """
    if source_count < 1 or tiles_per_source < 1 or budget < 1:
        raise ValueError('Source count, tile count, and budget must be positive')
    total = min(budget, source_count * tiles_per_source)
    rng = random.Random(seed)
    sources = list(range(source_count))
    rng.shuffle(sources)
    quota, remainder = divmod(total, source_count)
    positions = list(range(tiles_per_source))
    rng.shuffle(positions)
    offset = 0
    selected = {}
    for rank, index in enumerate(sources):
        count = quota + (rank < remainder)
        if count:
            # Continue around a shuffled grid across sources: when the overall
            # budget permits, every spatial position is represented at least once.
            selected[index] = sorted(positions[(offset+j) % tiles_per_source] for j in range(count))
            offset += count
    return dict(sorted(selected.items()))


def stage_dataset(root, dataset, recipe, train_budget=DEFAULT_TRAIN_TILES,
                  validation_budget=DEFAULT_VALIDATION_TILES, seed=42):
    root, dataset = Path(root), Path(dataset)
    files = sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(files) > 2000:
        raise ValueError('Maximum 2000 source images per run; curate a representative dataset first')
    print(f'Scanning and deduplicating {len(files)} source images...', flush=True)
    unique = {}
    source_shape = None
    for file_index, p in enumerate(files):
        if file_index and file_index % 100 == 0:
            print(f'Scanned {file_index}/{len(files)} source images', flush=True)
        # Read bounded chunks rather than allocating an entire encoded 4K image.
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        key = digest.hexdigest()
        if key in unique:
            continue
        try:
            with Image.open(p) as image:
                shape = [image.height, image.width]
                if image.width * image.height > 25_000_000:
                    raise ValueError('Image exceeds 25 megapixels')
                image.verify()
        except Exception as exc:
            raise ValueError(f'Unreadable image {p.name}: {exc}') from exc
        if source_shape is None:
            source_shape = shape
        if shape != source_shape:
            raise ValueError('All source images must have the same camera resolution')
        unique[key] = p
    files = list(unique.values())
    if len(files) < 30:
        raise ValueError('At least 30 distinct good images are required (100–500 recommended)')
    random.Random(seed).shuffle(files)
    nval = max(6, len(files) // 5)
    splits = {'normal': files[nval:], 'validation': files[:nval]}
    x0, y0, x1, y1 = roi_box(source_shape, recipe)
    boxes = list(tile_boxes((y1-y0, x1-x0), recipe))
    print(f'Dataset: {len(files)} distinct source images; {source_shape[1]} x {source_shape[0]}; '
          f'{len(boxes)} native-resolution tiles per image', flush=True)
    report = {'seed': seed, 'source_shape': source_shape, 'tiles_per_source': len(boxes), 'splits': {}}
    for name, paths in splits.items():
        budget = train_budget if name == 'normal' else validation_budget
        selection = select_tiles(len(paths), len(boxes), budget, seed + (name == 'validation'))
        selected_count = sum(map(len, selection.values()))
        total = len(paths) * len(boxes)
        print(f'{name}: staging {selected_count} / {total} tiles from '
              f'{len(selection)} / {len(paths)} sources (budget {budget}). '
              'Inference still inspects every tile.', flush=True)
        if len(selection) < len(paths):
            print(f'{name}: {len(paths)-len(selection)} sources are not sampled; '
                  'increase the tile budget or curate a smaller representative dataset for broader source coverage.', flush=True)
        target = root / 'dataset' / name
        target.mkdir(parents=True)
        source_records = []
        for i, p in enumerate(paths):
            indices = selection.get(i, [])
            source_records.append({'file': str(p), 'tiles': indices})
            if not indices:
                continue
            frame = cv2.imread(str(p))
            if frame is None:
                raise ValueError(f'Unreadable image: {p.name}')
            if list(frame.shape[:2]) != source_shape:
                raise ValueError(f'Image dimensions changed or cannot be decoded consistently: {p.name}')
            image, glare, _ = prepare(frame, recipe)
            if glare.mean() > recipe.max_glare:
                raise ValueError(f'Excess glare in {p.name}; recapture under controlled lighting')
            for j in indices:
                x, y, x1, y1 = boxes[j]
                # Preserve native resolution: sample crops, never resize whole 4K images.
                tile = image[y:y1, x:x1]
                if not cv2.imwrite(str(target / f'{i:05}_{j:03}.png'), tile):
                    raise OSError('Could not write staged training image')
            print(f'{name}: staged source {i+1}/{len(paths)}', flush=True)
        report['splits'][name] = {
            'available_sources': len(paths), 'selected_sources': len(selection),
            'available_tiles': total, 'selected_tiles': selected_count,
            'budget': budget, 'sources': source_records,
        }
    return report
