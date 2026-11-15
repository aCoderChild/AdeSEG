import numpy as np
import pytest
from modeling.lean_historical_usage_score import percentile_ranks

def test_percentiles_preserve_order_and_average_ties():
    assert np.allclose(percentile_ranks([4,1,1,3]),[1,1/6,1/6,2/3])
    assert np.allclose(percentile_ranks([5,5,5]),[.5,.5,.5])

def test_percentiles_reject_bad_values():
    with pytest.raises(ValueError):percentile_ranks([float('nan'),1])
    with pytest.raises(ValueError):percentile_ranks([1])
