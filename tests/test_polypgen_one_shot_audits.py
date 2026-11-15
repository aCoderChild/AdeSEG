import numpy as np
from scripts.audit_polypgen_one_shot_boxes import geometry
from scripts.audit_polypgen_one_shot_masks import metrics

def test_box_scoring_distinguishes_missing_and_empty_labels():
 gt=np.zeros((4,4),dtype=bool);gt[1:3,1:3]=True
 exact=geometry([1.,1.,3.,3.],gt)
 assert exact['detected'] and exact['box_iou']==1 and exact['box_truth_coverage']==1 and exact['labeled_empty_false_alarm'] is None
 missing=geometry(None,gt)
 assert not missing['detected'] and missing['box_iou']==0 and missing['box_truth_coverage']==0
 empty=np.zeros((4,4),dtype=bool)
 assert geometry(None,empty)['labeled_empty_false_alarm'] is False
 assert geometry([1.,1.,3.,3.],empty)['labeled_empty_false_alarm'] is True

def test_mask_scoring_reports_empty_false_alarms_separately():
 gt=np.zeros((4,4),dtype=bool);gt[1:3,1:3]=True
 pred=np.zeros_like(gt);pred[1,1]=True
 x=metrics(pred,gt)
 assert x['dice']==.4 and x['precision']==1 and x['recall']==.25 and x['false_alarm'] is None
 e=metrics(pred,np.zeros_like(gt))
 assert e['dice'] is None and e['false_alarm'] is True
