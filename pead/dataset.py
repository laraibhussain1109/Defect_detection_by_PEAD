"""Bounded, source-separated staging for high-resolution normal-image datasets."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random

import cv2
from PIL import Image

from .imaging import prepare_with_mask, clipped_fraction, roi_box, tile_boxes

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


def select_candidate_tiles(candidates, budget, seed=42):
    """Balance across sources without ever selecting a background-only crop."""
    if not candidates or any(not indices for indices in candidates):
        raise ValueError('Each source must contain at least one part tile; check the ROI and green mask preview')
    if all(indices == candidates[0] for indices in candidates):
        selected = select_tiles(len(candidates), len(candidates[0]), budget, seed)
        return {source: [candidates[source][index] for index in indices] for source, indices in selected.items()}
    rng = random.Random(seed)
    source_order = list(range(len(candidates)))
    rng.shuffle(source_order)
    pools = [list(indices) for indices in candidates]
    for pool in pools:
        rng.shuffle(pool)
    selected = {}
    used = 0
    for round_index in range(max(map(len, pools))):
        for source in source_order:
            if used == budget:
                return {key: sorted(value) for key, value in sorted(selected.items())}
            if round_index < len(pools[source]):
                selected.setdefault(source, []).append(pools[source][round_index])
                used += 1
    return {key: sorted(value) for key, value in sorted(selected.items())}


def write_quality_report(root, report):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root/'quality.tmp'
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    os.replace(temporary, root/'quality.json')


def stage_dataset(root, dataset, recipe, train_budget=DEFAULT_TRAIN_TILES,
                  validation_budget=DEFAULT_VALIDATION_TILES, seed=42):
    root, dataset = Path(root), Path(dataset)
    files = sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(files) > 2000:
        raise ValueError('Maximum 2000 source images per run; curate a representative dataset first')
    print(f'Scanning and deduplicating {len(files)} source images...', flush=True)
    unique = {}
    duplicates = []
    for file_index, p in enumerate(files):
        if file_index and file_index % 100 == 0:
            print(f'Scanned {file_index}/{len(files)} source images', flush=True)
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        key = digest.hexdigest()
        if key in unique:
            duplicates.append({'file':str(p),'duplicate_of':str(unique[key][0])})
            continue
        try:
            with Image.open(p) as image:
                shape = [image.height, image.width]
                if image.width * image.height > 25_000_000:
                    raise ValueError('Image exceeds 25 megapixels')
                image.verify()
        except Exception as exc:
            raise ValueError(f'Unreadable image {p.name}: {exc}') from exc
        unique[key] = (p, shape)

    # Assess every distinct source BEFORE assigning a train/holdout split or
    # spending the tile budget. Rejected frames never enter either dataset.
    source_shape = None
    boxes = None
    accepted = {}
    rejected = []
    quality_recipe = replace(recipe, clahe=False)
    print(f'Checking glare and visible part in {len(unique)} distinct images...', flush=True)
    for source_index, (path, header_shape) in enumerate(unique.values()):
        frame = cv2.imread(str(path))
        if frame is None or list(frame.shape[:2]) != header_shape:
            raise ValueError(f'Cannot decode image consistently: {path.name}')
        _, glare, _, mask = prepare_with_mask(frame, quality_recipe)
        fraction = clipped_fraction(glare, mask)
        if fraction > recipe.max_glare:
            rejected.append({'file':str(path),'reason':'excess_glare','glare_fraction':fraction,
                             'limit':recipe.max_glare})
            print(f'REJECTED glare: {path.name} ({fraction:.2%} > {recipe.max_glare:.2%}); '
                  'skipping image, original preserved', flush=True)
            continue
        if mask.mean() < recipe.min_part_fraction:
            raise ValueError(f'No part found in {path.name}: check the ROI and green-background mask preview')
        if source_shape is None:
            source_shape = header_shape
            x0,y0,x1,y1 = roi_box(source_shape,recipe)
            boxes = list(tile_boxes((y1-y0,x1-x0),recipe))
        elif header_shape != source_shape:
            raise ValueError('All accepted source images must have the same camera resolution')
        if recipe.green_background:
            integral = cv2.integral(mask.astype('uint8'))
            eligible = []
            for index,(x,y,x1,y1) in enumerate(boxes):
                pixels = int(integral[y1,x1])-int(integral[y,x1])-int(integral[y1,x])+int(integral[y,x])
                if pixels / ((x1-x)*(y1-y)) >= recipe.min_part_fraction:
                    eligible.append(index)
            if not eligible:
                raise ValueError(f'No usable part tiles in {path.name}: check the ROI and green mask preview')
        else:
            eligible = list(range(len(boxes)))
        stat = path.stat()
        accepted[path] = {'tiles':eligible,'glare_fraction':fraction,
                          'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if (source_index+1) % 25 == 0 or source_index+1 == len(unique):
            print(f'Quality checked {source_index+1}/{len(unique)} distinct sources; '
                  f'accepted {len(accepted)}, rejected {len(rejected)} glare', flush=True)
    summary = {'scanned_files':len(files),'distinct_sources':len(unique),
               'accepted_sources':len(accepted),'rejected_sources':len(rejected),
               'duplicate_files':len(duplicates)}
    quality = {**summary,'glare_limit':recipe.max_glare,'rejected':rejected,'duplicates':duplicates,
               'accepted':[{'file':str(path),'glare_fraction':item['glare_fraction']} for path,item in accepted.items()]}
    write_quality_report(root,quality)
    print(f'Quality summary: {len(accepted)} accepted, {len(rejected)} glare-rejected, '
          f'{len(duplicates)} duplicate files excluded. Source files are unchanged.', flush=True)
    if len(accepted) < 30:
        raise ValueError(f'Insufficient accepted images after quality filtering: {len(accepted)} accepted, '
                         f'{len(rejected)} rejected for glare, {len(duplicates)} duplicates. '
                         'At least 30 distinct accepted good images are required; see quality.json.')

    paths = list(accepted)
    random.Random(seed).shuffle(paths)
    nval = max(6,len(paths)//5)
    splits = {'normal':paths[nval:],'validation':paths[:nval]}
    print(f'Dataset: {len(paths)} accepted distinct sources; {source_shape[1]} x {source_shape[0]}; '
          f'{len(boxes)} native-resolution grid tiles per image', flush=True)
    report = {'seed':seed,'source_shape':source_shape,'tiles_per_source':len(boxes),
              'quality':summary,'splits':{}}
    for name, paths in splits.items():
        requested_budget = train_budget if name == 'normal' else validation_budget
        # A tile limit must not silently drop an accepted source. Preserve
        # bounded per-image sampling, with at least one crop per source.
        budget = max(requested_budget,len(paths))
        if budget > requested_budget:
            print(f'{name}: increasing tile budget {requested_budget} -> {budget} '
                  'to include every accepted source', flush=True)
        candidates = [accepted[path]['tiles'] for path in paths]
        selection = select_candidate_tiles(candidates,budget,seed+(name=='validation'))
        selected_count = sum(map(len,selection.values()))
        total = sum(map(len,candidates))
        print(f'{name}: staging {selected_count} / {total} tiles from '
              f'{len(selection)} / {len(paths)} accepted sources (effective budget {budget}). '
              'Inference inspects every part tile.', flush=True)
        target = root/'dataset'/name
        target.mkdir(parents=True)
        masks_target = root/'dataset_masks'/name
        if recipe.green_background:
            masks_target.mkdir(parents=True)
        records = []
        for i,path in enumerate(paths):
            indices = selection[i]
            records.append({'file':str(path),'tiles':indices,'eligible_tiles':len(candidates[i])})
            # Captures are unique files. Detect in-place edits rather than
            # training on different pixels from those that passed preflight.
            current = path.stat()
            previous = accepted[path]
            if current.st_size != previous['size'] or current.st_mtime_ns != previous['mtime_ns']:
                raise ValueError(f'Image changed after quality filtering: {path.name}; keep the source folder unchanged and rerun')
            frame = cv2.imread(str(path))
            if frame is None or list(frame.shape[:2]) != source_shape:
                raise ValueError(f'Cannot decode accepted image consistently: {path.name}')
            image,_,_,mask = prepare_with_mask(frame,recipe)
            for j in indices:
                x,y,x1,y1 = boxes[j]
                if not cv2.imwrite(str(target/f'{i:05}_{j:03}.png'),image[y:y1,x:x1]):
                    raise OSError('Could not write staged training image')
                if recipe.green_background:
                    if not cv2.imwrite(str(masks_target/f'{i:05}_{j:03}.png'),mask[y:y1,x:x1].astype('uint8')*255):
                        raise OSError('Could not write training part mask')
            print(f'{name}: staged accepted source {i+1}/{len(paths)}',flush=True)
        report['splits'][name] = {'available_sources':len(paths),'selected_sources':len(selection),
            'available_tiles':total,'selected_tiles':selected_count,
            'background_tiles_excluded':len(paths)*len(boxes)-total,
            'budget':requested_budget,'effective_budget':budget,'sources':records}
    return report
