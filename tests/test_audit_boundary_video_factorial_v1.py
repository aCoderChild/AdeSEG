import json
import numpy as np
import pytest

from scripts import audit_boundary_video_factorial_v1 as audit
from scripts import score_boundary_video_factorial_v1 as primary


def test_independent_metric_matches_primary_for_synthetic_shapes():
    rng = np.random.default_rng(2026)
    for kind in ('empty','miss','edge','random'):
        pred = np.zeros((31,43),dtype=bool); truth=pred.copy()
        if kind == 'miss': truth[4:8,6:11] = True
        if kind == 'edge': pred[:10,:14]=True; truth[2:12,3:15]=True
        if kind == 'random': pred=rng.random(pred.shape)>.7; truth=rng.random(truth.shape)>.8
        a,b=audit.metrics(pred,truth),primary.metric(pred,truth)
        assert set(a)==set(b)
        assert all(audit.same(a[k],b[k]) for k in a)
        cached = audit.metrics(pred, truth, audit.boundary_target(truth))
        assert all(audit.same(cached[k], b[k]) for k in cached)


def test_summary_and_pair_comparisons_handle_empty_and_undefined():
    empty=np.zeros((9,9),dtype=bool); truth=empty.copy();truth[2:5,2:5]=True
    rows=[{'post_prompt':True, 'A':audit.metrics(empty,truth), 'B':audit.metrics(truth,truth)},
          {'post_prompt':True, 'A':audit.metrics(empty,empty), 'B':audit.metrics(truth,empty)}]
    assert audit.summarize(rows,'A',True)['mean_present_precision_when_defined'] is None
    assert audit.summarize(rows,'B',True)['false_alarms'] == 1
    assert audit.paired(rows,'B','A')['mean_present_dice_difference'] == 1.0


def test_unfinished_study_fails_before_label_read(tmp_path,monkeypatch):
    (tmp_path/'run_manifest.json').write_text(json.dumps({}))
    (tmp_path/'run_map.json').write_text(json.dumps({}))
    def forbidden(*args):
        raise AssertionError('label opened before complete study')
    monkeypatch.setattr(audit,'label',forbidden)
    with pytest.raises(FileNotFoundError): audit.audit(tmp_path)
