import numpy as np
from PIL import Image
import pytest

from datasets.vos_benchmarks import author_mask_io, inventory


def frames(root,folder,count):
    directory=root/folder/'video';directory.mkdir(parents=True)
    for i in range(count):
        Image.fromarray(np.zeros((8,10,3),dtype=np.uint8)).save(directory/f'{i:05d}.jpg')


def test_sav_preserves_object_zero_overlap_and_full_rate_frames(tmp_path):
    frames(tmp_path,'JPEGImages_24fps',9)
    (tmp_path/'sav_val.txt').write_text('video\n')
    for object_id in (0,1):
        directory=tmp_path/f'Annotations_6fps/video/{object_id:03d}';directory.mkdir(parents=True)
        for frame in (0,4,8):
            mask=np.zeros((8,10),dtype=np.uint8);mask[2:6,3:7]=255
            Image.fromarray(mask).save(directory/f'{frame:05d}.png')
    result=inventory(tmp_path,'sav','val')
    assert result['input_frames']==9 and result['object_count']==2
    assert result['videos'][0]['object_ids']==[0,1]
    assert result['videos'][0]['annotations']['0']==['00000','00004','00008']
    masks,_=author_mask_io()['load_masks_from_dir'](str(tmp_path/'Annotations_6fps'),'video','00000',True)
    assert masks[0][3,4] and masks[1][3,4]


def test_davis_preserves_indexed_ids_and_ignores_void(tmp_path):
    frames(tmp_path,'JPEGImages/480p',3)
    listing=tmp_path/'ImageSets/2017';listing.mkdir(parents=True)
    (listing/'val.txt').write_text('video\n')
    directory=tmp_path/'Annotations/480p/video';directory.mkdir(parents=True)
    for frame in range(3):
        mask=np.zeros((8,10),dtype=np.uint8)
        mask[1:3,1:3]=7;mask[4:6,5:8]=42;mask[0,0]=255
        Image.fromarray(mask).save(directory/f'{frame:05d}.png')
    result=inventory(tmp_path,'davis2017','val')
    assert result['videos'][0]['object_ids']==[7,42]
    assert not result['per_object_png'] and result['input_frames']==3
    (directory/'00002.png').unlink()
    with pytest.raises(ValueError,match='every input frame'):
        inventory(tmp_path,'davis2017','val')


def test_inventory_rejects_unsafe_official_list_entry(tmp_path):
    (tmp_path/'sav_test.txt').write_text('../video\n')
    with pytest.raises(ValueError,match='official video list'):
        inventory(tmp_path,'sav','test')
