"""Pairwise index-matched swaps for frozen cross-trajectory reads."""
from modeling.lean_cross_trajectory_blocks import swap_block

PAIRS = {'first_middle': ('first', 'middle'),
         'first_last': ('first', 'last'),
         'middle_last': ('middle', 'last')}

def swap_pair(recipient, donor, pair):
    first, second = PAIRS[pair]
    return swap_block(swap_block(recipient, donor, first), donor, second)
