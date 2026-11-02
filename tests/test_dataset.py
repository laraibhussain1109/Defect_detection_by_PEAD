import cv2
import numpy as np
import pytest

from pead.dataset import select_tiles, stage_dataset, training_settings
from pead.imaging import Recipe, tile_boxes


def test_4k_plan_exceeds_old_limit_but_is_bounded_and_spatially_complete():
    grid = list(tile_boxes((2160, 3840), Recipe()))
    assert len(grid) == 60
    assert 80 * len(grid) > 4000
    selected = select_tiles(80, len(grid), 512)
    assert sum(map(len, selected.values())) == 512
    assert len(selected) == 80
    assert max(map(len, selected.values())) - min(map(len, selected.values())) <= 1
    assert set().union(*map(set, selected.values())) == set(range(60))
    assert selected == select_tiles(80, len(grid), 512)
    assert selected != select_tiles(80, len(grid), 512, seed=43)
    assert all(len(indices) == len(set(indices)) for indices in selected.values())


def test_budget_smaller_than_source_count_is_bounded_and_reportable():
    selected = select_tiles(1600, 60, 128)
    assert len(selected) == 128
    assert sum(map(len, selected.values())) == 128
    assert set().union(*map(set, selected.values())) == set(range(60))
    assert select_tiles(3, 2, 100) == {0: [0, 1], 1: [0, 1], 2: [0, 1]}


def test_4k_small_tiles_cover_frame_without_old_128_tile_failure():
    boxes = list(tile_boxes((2160, 3840), Recipe(tile_size=256)))
    assert len(boxes) > 128
    coverage = np.zeros((2160, 3840), np.uint8)
    for x, y, x1, y1 in boxes:
        coverage[y:y1, x:x1] = 1
    assert coverage.all()


def test_settings_work_for_legacy_jobs_and_quick_check():
    job = {'epochs': 100, 'coreset': .01}
    assert training_settings(job)['train_tile_budget'] == 512
    quick = training_settings({**job, 'quick_check': True})
    assert quick == {'epochs': 1, 'coreset': .001, 'quick_check': True,
                     'train_tile_budget': 128, 'validation_tile_budget': 64}
    with pytest.raises(ValueError, match='tile budget'):
        training_settings({**job, 'train_tile_budget': 4000})


def test_actual_4k_dataset_that_previously_failed_stages_with_default_budgets(tmp_path):
    source = tmp_path / 'good'
    source.mkdir()
    frame = np.empty((2160, 3840, 3), np.uint8)
    for i in range(100):
        frame.fill(40+i)
        assert cv2.imwrite(str(source / f'{i:03}.jpg'), frame)
    # Exact duplicates must not inflate source counts or leak across the split.
    (source / 'duplicate.jpg').write_bytes((source / '000.jpg').read_bytes())
    report = stage_dataset(tmp_path/'run', source, Recipe())
    assert report['source_shape'] == [2160, 3840]
    train, val = report['splits']['normal'], report['splits']['validation']
    assert (train['available_sources'], val['available_sources']) == (80, 20)
    assert (train['available_tiles'], val['available_tiles']) == (4800, 1200)
    assert (train['selected_tiles'], val['selected_tiles']) == (512, 128)
    assert (train['selected_sources'], val['selected_sources']) == (80, 20)
    assert not ({r['file'] for r in train['sources']} & {r['file'] for r in val['sources']})
    outputs = list((tmp_path/'run'/'dataset'/'normal').glob('*.png'))
    assert len(outputs) == 512
    assert len(list((tmp_path/'run'/'dataset'/'validation').glob('*.png'))) == 128
    assert cv2.imread(str(outputs[0])).shape == (512, 512, 3)
    assert {p.suffix for p in source.iterdir()} == {'.jpg'}


def test_staging_rejects_mixed_resolutions_before_writing_tiles(tmp_path):
    source = tmp_path/'good'
    source.mkdir()
    for i, size in enumerate([256, 512]):
        cv2.imwrite(str(source/f'{i}.png'), np.full((size,size,3), 90, np.uint8))
    with pytest.raises(ValueError, match='same camera resolution'):
        stage_dataset(tmp_path/'run', source, Recipe())
    assert not (tmp_path/'run'/'dataset').exists()
