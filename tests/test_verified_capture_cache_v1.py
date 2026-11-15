import torch
import pytest

from scripts.memory_token_io import digest
from scripts.verified_capture_cache_v1 import VerifiedCaptureCache


def item(tmp_path, number):
    feature = tmp_path/f'feature{number}'; feature.write_bytes(f'f{number}'.encode())
    snapshot = tmp_path/f'snapshot{number}'; snapshot.write_bytes(f's{number}'.encode())
    return {'sequence': 'seq8', 'frame': number, 'feature': str(feature), 'feature_sha256': digest(feature),
            'snapshot': str(snapshot), 'snapshot_sha256': digest(snapshot)}


def test_bounded_cache_hits_evicts_and_preserves_cpu_tensors(tmp_path):
    calls = []
    def loader(row):
        calls.append(row['frame'])
        return (torch.tensor([float(row['frame'])]),), (1, 1)
    cache = VerifiedCaptureCache(8, loader)  # two float32 elements
    a, b, c = (item(tmp_path, x) for x in range(3))
    first, _ = cache.get(a, 'cpu'); first[0].fill_(99)  # caller cannot alter cached CPU value
    assert cache.get(a, 'cpu')[0][0].item() == 0
    cache.get(b, 'cpu'); cache.get(c, 'cpu'); cache.get(a, 'cpu')
    assert calls == [0, 1, 2, 0]
    assert cache.stats()['hits'] == 1 and cache.stats()['peak_bytes'] <= 8
    cache.verify_boundary(final=True)


def test_cache_fails_when_capture_mutates_on_hit_or_before_checkpoint(tmp_path):
    row = item(tmp_path, 0)
    cache = VerifiedCaptureCache(8, lambda _: ((torch.ones(1),), (1, 1)))
    cache.get(row, 'cpu')
    (tmp_path/'snapshot0').write_bytes(b'changed')
    with pytest.raises(ValueError, match='metadata changed'):
        cache.get(row, 'cpu')
    with pytest.raises(ValueError, match='metadata changed'):
        cache.verify_boundary(final=True)


def test_cache_rejects_wrong_digest_on_initial_load(tmp_path):
    row = item(tmp_path, 0); row['feature_sha256'] = '0' * 64
    cache = VerifiedCaptureCache(8, lambda _: ((torch.ones(1),), (1, 1)))
    with pytest.raises(ValueError, match='hash differs'):
        cache.get(row, 'cpu')


def test_enabled_cache_preserves_deterministic_update_sequence(tmp_path):
    rows = [item(tmp_path, i) for i in range(3)]
    def run(limit):
        cache = VerifiedCaptureCache(limit, lambda row: ((torch.tensor([float(row['frame'] + 1)]),), (1, 1)))
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(.25)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        losses = []
        for index in (0, 1, 0, 2, 0, 1):
            (value,), _ = cache.get(rows[index], 'cpu')
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(value) - (value * 2)) ** 2).mean()
            loss.backward(); optimizer.step(); losses.append(loss.item())
        cache.verify_boundary(final=True)
        return model.weight.detach().clone(), losses, cache.stats()
    uncached, losses_a, stats_a = run(0)
    cached, losses_b, stats_b = run(8)
    assert torch.equal(uncached, cached) and losses_a == losses_b
    assert stats_a['hits'] == 0 and stats_b['hits'] > 0
