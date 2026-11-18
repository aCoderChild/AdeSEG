"""Check scientific public names are identical to the historical checkpoint class."""
from modeling.feature_aggregation import MultistageFeatureAggregator
from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1

def test_model_class_identity_for_video_routing():
    assert MultistageFeatureAggregator is MultiscaleChannelValueRefinerV1
