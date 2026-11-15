"""Reuse the native-cache diagnostic runner with pinned quarter-budget matchers."""
from pathlib import Path
import json
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.author_token_matching import use_matching
from modeling.pitome_matching import use_pitome
from modeling.region_constrained_tome import NativeRegionConstrainedToMe
from scripts import run_author_matching_diagnostic as reader
from scripts import run_cached_author_matching_diagnostic as cached


def make_protected_reader(protocol):
    """Bind the frozen protocol once, outside the per-target replay loop."""
    if protocol['merge_divisor'] != 4 or protocol['margin'] != .5:
        raise ValueError('The protection diagnostic has one frozen budget/margin')

    def manager(*args, **kwargs):
        return NativeRegionConstrainedToMe(*args, **kwargs, merge_divisor=4)

    def matching(strategy):
        return use_pitome(Path(protocol['pitome_repository'])) if strategy == 'pitome' else use_matching(strategy)

    def protected_read(*args, **kwargs):
        with patch.object(reader, 'NativeRegionConstrainedToMe', manager), \
             patch.object(reader, 'use_matching', matching):
            return reader.test_read(*args, **kwargs)

    return protected_read


if __name__ == '__main__':
    plan_argument = sys.argv[sys.argv.index('--plan') + 1]
    protocol = json.loads(Path(plan_argument).read_text())['protocol']
    with patch.object(cached, 'test_read', make_protected_reader(protocol)):
        cached.main()
