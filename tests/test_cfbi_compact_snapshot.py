import pytest
import torch
from modeling.cfbi_compact_snapshot import compact_training_read


def test_compact_preserves_selected_memory_and_all_pointers_without_mutation():
    def entry(n):
        return {'maskmem_features':torch.tensor([n]),'pred_masks':torch.tensor([n]),
                'maskmem_pos_enc':[torch.tensor([n])],'obj_ptr':torch.tensor([n])}
    sample={'frame':5,'positions':[torch.ones(3),torch.ones(2),torch.ones(1)],
            'output_dict':{'cond_frame_outputs':{0:entry(0)},
                           'non_cond_frame_outputs':{i:entry(i) for i in range(1,5)}}}
    compact=compact_training_read(sample,[0,3,4])
    assert [p.numel() for p in compact['positions']]==[0,0,1]
    assert compact['output_dict']['non_cond_frame_outputs'][1]['maskmem_features'] is None
    assert compact['output_dict']['non_cond_frame_outputs'][1]['obj_ptr'].item()==1
    assert compact['output_dict']['non_cond_frame_outputs'][3]['maskmem_features'].item()==3
    assert sample['output_dict']['non_cond_frame_outputs'][1]['maskmem_features'].item()==1
    with pytest.raises(ValueError):compact_training_read(sample,[0,5])
    with pytest.raises(ValueError):compact_training_read(sample,[0,0])
