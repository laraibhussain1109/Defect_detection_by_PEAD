import cv2
import numpy as np
import pytest
from pead.imaging import Recipe, prepare, tiles, overlay


def test_tiles_cover_edges_without_losing_native_pixels():
    image=np.zeros((777,1133,3),np.uint8)
    count=np.zeros(image.shape[:2],np.uint8)
    for (x,y,x1,y1),tile in tiles(image,Recipe(tile_size=512)):
        assert tile.shape[:2]==(y1-y,x1-x)
        count[y:y1,x:x1]+=1
    assert count.min()>=1
    assert count.max()>1


def test_recipe_rejects_out_of_bounds_roi():
    with pytest.raises(ValueError): Recipe(roi=(.8,0,.5,1))


def test_glare_does_not_become_a_normal_or_anomaly_prediction():
    frame=np.full((64,64,3),255,np.uint8)
    recipe=Recipe(min_area=4)
    image,glare,box=prepare(frame,recipe)
    annotated,result=overlay(frame,np.ones((64,64),np.float32)*2,glare,box,recipe)
    assert result['decision']=='UNINSPECTABLE'
    assert result['regions']==[]
    assert result['glare_fraction']==1


def test_anomaly_localizes_in_roi_coordinates_and_ignores_speckles():
    frame=np.full((100,200,3),80,np.uint8)
    recipe=Recipe(roi=(.5,0,.5,1),min_area=10)
    _,glare,box=prepare(frame,recipe)
    score=np.zeros((100,100),np.float32)
    score[20:30,40:50]=1.5
    score[1,1]=2
    _,result=overlay(frame,score,glare,box,recipe)
    assert result['decision']=='ANOMALY'
    assert len(result['regions'])==1
    assert result['regions'][0]['x']==40
    assert result['regions'][0]['area']==100


def test_doubtful_and_normal_use_fixed_thresholds():
    frame=np.full((50,50,3),80,np.uint8)
    recipe=Recipe(min_area=4)
    _,glare,box=prepare(frame,recipe)
    for value,decision in [(.1,'NORMAL'),(.8,'REVIEW'),(1.2,'ANOMALY')]:
        _,result=overlay(frame,np.full((50,50),value,np.float32),glare,box,recipe)
        assert result['decision']==decision
