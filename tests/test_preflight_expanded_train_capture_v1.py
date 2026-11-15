from scripts.preflight_expanded_train_capture_v1 import TARGETS,digest_pipeline


def test_fixed_label_free_target_schedule():
    for name,start,count in (('seq6',23,91),('seq13',7,250)):
        expected=sorted({start+1+round((count-1-(start+1))*i/9) for i in range(10)})
        assert TARGETS[name]==expected and len(expected)==10


def test_pipeline_hash_is_order_independent():
    assert digest_pipeline({'a':1,'b':2})==digest_pipeline({'b':2,'a':1})
    assert digest_pipeline({'a':1})!=digest_pipeline({'a':2})
