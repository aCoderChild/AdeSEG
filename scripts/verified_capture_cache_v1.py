"""Bounded CPU cache for hash-verified fixed-read tensors.

Each item is verified on load, file metadata is checked on cache reuse and at
epoch boundaries, and all used bytes are rehashed before a checkpoint is
accepted. This does not claim continuous filesystem immutability.
"""
from collections import OrderedDict
from pathlib import Path

from scripts.memory_token_io import digest


def signature(item):
    result = []
    for name in ('feature', 'snapshot'):
        path = Path(item[name]).resolve(strict=True)
        stat = path.stat()
        if not path.is_file():
            raise ValueError('Capture is not a regular file')
        result.append((str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(result)


class VerifiedCaptureCache:
    def __init__(self, limit_bytes, loader):
        if type(limit_bytes) is not int or not 0 <= limit_bytes <= 1024**3:
            raise ValueError('CPU cache limit must be between 0 and 1 GiB')
        self.limit = limit_bytes
        self.loader = loader
        self.cache = OrderedDict()
        self.seen = {}
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.final_verified = False

    @staticmethod
    def key(item):
        return (item['sequence'], item['frame'], item['feature'], item['feature_sha256'],
                item['snapshot'], item['snapshot_sha256'])

    def get(self, item, device):
        self.final_verified = False
        key = self.key(item); before = signature(item)
        if key in self.seen and before != self.seen[key][1]:
            raise ValueError('Capture file metadata changed after validation')
        if key in self.cache:
            self.hits += 1
            tensors, shape, size = self.cache.pop(key)
            self.cache[key] = (tensors, shape, size)
        else:
            self.misses += 1
            tensors, shape = self.loader(item)
            if any(digest(item[name]) != item[name + '_sha256'] for name in ('feature', 'snapshot')):
                raise ValueError('Capture file hash differs after loading')
            if signature(item) != before:
                raise ValueError('Capture file changed while loading')
            if not all(t.device.type == 'cpu' for t in tensors):
                raise ValueError('Cache loader must return CPU tensors')
            size = sum(t.numel() * t.element_size() for t in tensors)
            self.seen[key] = (dict(item), before)
            if size <= self.limit:
                while self.cache and self.current_bytes + size > self.limit:
                    _, (_, _, evicted) = self.cache.popitem(last=False)
                    self.current_bytes -= evicted
                self.cache[key] = (tensors, shape, size)
                self.current_bytes += size
                self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return tuple(t.to(device, copy=(device == 'cpu')) for t in tensors), shape

    def verify_boundary(self, *, final=False):
        for item, expected in self.seen.values():
            if signature(item) != expected:
                raise ValueError('Capture file metadata changed during training')
            if final:
                for name in ('feature', 'snapshot'):
                    if digest(item[name]) != item[name + '_sha256']:
                        raise ValueError('Capture file hash changed during training')
                if signature(item) != expected:
                    raise ValueError('Capture file changed during final verification')
        if final:
            self.final_verified = True

    def stats(self):
        return {'limit_bytes': self.limit, 'peak_bytes': self.peak_bytes,
                'hits': self.hits, 'misses': self.misses, 'unique_reads': len(self.seen),
                'final_sha256_verified': self.final_verified,
                'scope': 'Tensor bytes only, not total RSS. Validated on load; metadata checked on reuse and epoch boundaries; bytes rehashed before checkpoint. Speed and hit rate are measured at runtime.'}
