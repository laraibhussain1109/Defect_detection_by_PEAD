import hashlib
import json

import cv2
import numpy as np
import pytest

from pead.dataset import stage_dataset
from pead.imaging import Recipe


def write_sources(folder, good=30, glare=3, green=False):
    folder.mkdir()
    for i in range(good):
        image=np.full((512,512,3),40+i,np.uint8)
        if green:
            image[:]=(20,140,20)
            image[128:384,128:384]=40+i
        cv2.imwrite(str(folder/f'good_{i:03}.png'),image)
    for i in range(glare):
        image=np.full((512,512,3),120,np.uint8)
        if green:
            image[:]=(20,140,20)
            image[128:384,128:384]=255
            image[128,128]=250-i
        else:
            image[:]=255
            image[0,0]=250-i
        cv2.imwrite(str(folder/f'bad_{i:03}.png'),image)


@pytest.mark.parametrize('green',[False,True])
def test_glare_images_are_skipped_before_split_and_originals_unchanged(tmp_path,green):
    source=tmp_path/'good';write_sources(source,green=green)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    run=tmp_path/'run'
    recipe=Recipe(green_background=green,tile_size=256)
    report=stage_dataset(run,source,recipe,train_budget=24,validation_budget=6)
    quality=json.loads((run/'quality.json').read_text())
    assert quality['accepted_sources']==30 and quality['rejected_sources']==3
    assert len(quality['accepted'])==30 and len(quality['rejected'])==3
    assert all(item['reason']=='excess_glare' for item in quality['rejected'])
    used={record['file'] for split in report['splits'].values() for record in split['sources']}
    assert len(used)==30 and all('good_' in path for path in used)
    assert not used & {item['file'] for item in quality['rejected']}
    assert report['splits']['normal']['available_sources']==24
    assert report['splits']['validation']['available_sources']==6
    assert len(list((run/'dataset'/'normal').glob('*.png')))==24
    assert len(list((run/'dataset'/'validation').glob('*.png')))==6
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}


def test_first_rejected_frame_does_not_set_the_accepted_resolution(tmp_path):
    source=tmp_path/'good';write_sources(source,glare=0)
    cv2.imwrite(str(source/'000_glare_different_size.png'),np.full((1024,1024,3),255,np.uint8))
    report=stage_dataset(tmp_path/'run',source,Recipe(),24,6)
    assert report['source_shape']==[512,512]
    assert report['quality']['rejected_sources']==1


def test_budget_expands_to_include_all_accepted_sources(tmp_path):
    source=tmp_path/'good';write_sources(source,good=40,glare=2)
    report=stage_dataset(tmp_path/'run',source,Recipe(tile_size=256),24,6)
    for split,expected in [('normal',32),('validation',8)]:
        data=report['splits'][split]
        assert data['available_sources']==data['selected_sources']==expected
        assert data['effective_budget']==expected
        assert data['selected_tiles']==expected
        assert all(record['tiles'] for record in data['sources'])


def test_too_few_accepted_images_reports_counts_after_all_glare_checks(tmp_path):
    source=tmp_path/'good';write_sources(source,good=29,glare=4)
    run=tmp_path/'run'
    with pytest.raises(ValueError,match='29 accepted, 4 rejected for glare'):
        stage_dataset(run,source,Recipe(),24,6)
    quality=json.loads((run/'quality.json').read_text())
    assert quality['accepted_sources']==29 and quality['rejected_sources']==4
    assert not (run/'dataset').exists()


def test_all_glare_has_summary_instead_of_first_image_error(tmp_path):
    source=tmp_path/'good';write_sources(source,good=0,glare=30)
    run=tmp_path/'run'
    with pytest.raises(ValueError,match='0 accepted, 30 rejected for glare'):
        stage_dataset(run,source,Recipe(),24,6)
    quality=json.loads((run/'quality.json').read_text())
    assert quality['rejected_sources']==30 and not quality['accepted']
    assert not (run/'dataset').exists()
