import torch
from scripts.verify_rde_cross_pipeline import capture_logits


def test_common_logit_observation_preserves_generator_and_restores_method():
    class Model:
        def __init__(self):self.closed=False
        def propagate_in_video(self):
            try:
                for i in range(2):yield i,[1],torch.full((1,1,2,2),float(i))
            finally:self.closed=True
    model=Model();original=model.propagate_in_video
    with capture_logits(model) as rows:
        outputs=list(model.propagate_in_video())
    assert model.propagate_in_video==original and model.closed
    assert [r['frame'] for r in rows]==[0,1] and all(r['shape']==[1,1,2,2] for r in rows)
    assert rows[0]['raw_sha256']!=rows[1]['raw_sha256']
    assert torch.equal(outputs[1][2],torch.ones(1,1,2,2))
