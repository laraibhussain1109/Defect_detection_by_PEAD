import cv2
import numpy as np
import pytest

from pead.imaging import Recipe, part_mask, prepare_with_mask, overlay
from pead.model import Hybrid
from pead.dataset import stage_dataset, select_candidate_tiles


def scene(size=512):
    image=np.full((size,size,3),(25,155,25),np.uint8)
    image[size//4:3*size//4,size//4:3*size//4]=130
    return image


def test_green_backdrop_excluded_but_internal_green_hole_is_kept():
    image=scene()
    image[250:263,250:263]=(25,155,25)  # about 1 mm at the user's field of view
    recipe=Recipe(green_background=True)
    mask=part_mask(image,recipe)
    assert not mask[20,20]
    assert mask[200,200]
    assert mask[255,255], 'An enclosed green-visible blowhole must remain inspectable'
    prepared,glare,box,part=prepare_with_mask(image,recipe)
    assert np.all(prepared[~part]==127)
    assert np.array_equal(prepared[250:263,250:263],image[250:263,250:263])
    assert box==(0,0,512,512) and not glare.any()


def test_background_anomaly_scores_do_not_create_part_defects():
    image=scene()
    recipe=Recipe(green_background=True,min_area=4)
    _,glare,box,part=prepare_with_mask(image,recipe)
    score=np.zeros((512,512),np.float32)
    score[~part]=10
    out,result=overlay(image,score,glare,box,recipe)
    assert result['decision']=='NORMAL'
    assert result['regions']==[]
    assert np.array_equal(out[20:30,20:30],image[20:30,20:30])
    score[250:263,250:263]=2
    _,result=overlay(image,score,glare,box,recipe)
    assert result['decision']=='ANOMALY'
    assert result['regions'][0]['area']==169


def test_empty_green_view_is_no_part_and_never_calls_models():
    image=np.full((512,512,3),(25,155,25),np.uint8)
    model=Hybrid.__new__(Hybrid)
    model.recipe=Recipe(green_background=True)
    model.meta={'source_shape':[512,512]}
    def never(_):
        raise AssertionError('No model inference should run on empty background')
    model.tile_maps=never
    score,glare,box=model.predict(image)
    _,result=overlay(image,score,glare,box,model.recipe)
    assert result['decision']=='NO_PART'
    assert result['score']==0 and not score.any()


def test_glare_ratio_is_measured_on_part_not_diluted_by_background():
    image=scene()
    image[128:256,128:384]=255
    recipe=Recipe(green_background=True,max_glare=.2)
    _,glare,box,_=prepare_with_mask(image,recipe)
    _,result=overlay(image,np.zeros((512,512),np.float32),glare,box,recipe)
    assert result['glare_fraction']>.5
    assert result['decision']=='UNINSPECTABLE'


def test_masked_staging_drops_background_tiles_and_saves_native_masks(tmp_path):
    source=tmp_path/'good';source.mkdir()
    for i in range(30):
        image=np.full((1024,1024,3),(25,155,25),np.uint8)
        image[400:624,400:624]=80+i
        cv2.imwrite(str(source/f'{i:02}.png'),image)
    recipe=Recipe(green_background=True,tile_size=256,min_area=4)
    report=stage_dataset(tmp_path/'run',source,recipe,train_budget=24,validation_budget=6)
    normal,validation=report['splits']['normal'],report['splits']['validation']
    assert normal['background_tiles_excluded']>0
    assert normal['available_tiles']<24*report['tiles_per_source']
    assert normal['selected_tiles']==24 and validation['selected_tiles']==6
    assert not ({r['file'] for r in normal['sources']} & {r['file'] for r in validation['sources']})
    for split in ['normal','validation']:
        for path in (tmp_path/'run'/'dataset'/split).glob('*.png'):
            image=cv2.imread(str(path))
            mask=cv2.imread(str(tmp_path/'run'/'dataset_masks'/split/path.name),0)>0
            assert mask.mean()>=recipe.min_part_fraction
            assert np.all(image[~mask]==127)
            assert image.shape==(256,256,3)


def test_variable_candidate_selection_never_chooses_background():
    candidates=[[3,8],[1],[2,4,9]]
    selected=select_candidate_tiles(candidates,5)
    assert sum(map(len,selected.values()))==5
    assert all(set(indices)<=set(candidates[source]) for source,indices in selected.items())
    assert selected==select_candidate_tiles(candidates,5)


def test_no_part_dataset_has_actionable_error(tmp_path):
    source=tmp_path/'good';source.mkdir()
    for i in range(30):
        image=np.full((128,128,3),(10,110+i,10),np.uint8)
        cv2.imwrite(str(source/f'{i}.png'),image)
    with pytest.raises(ValueError,match='No part found'):
        stage_dataset(tmp_path/'run',source,Recipe(green_background=True),24,6)
