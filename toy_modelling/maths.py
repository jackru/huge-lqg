"""
Maths functions for toy modelling
"""
import numpy as np
from scipy.special import comb


def binom_row(N, k):
    outlist = []
    for i in range(k, N):
        outlist.insert(0, comb(i, k, exact=True))
    return np.array(outlist)


def nth_combination(from_N, choose_k, nth_comb):
    assert nth_comb < comb(from_N, choose_k), "Index out of range"
    items_left = from_N
    index = nth_comb
    min_pick = 0
    outlist = []
    for picks_left in range(choose_k, 0, -1):
        increment_indices = binom_row(items_left, picks_left - 1).cumsum()
        pick = np.searchsorted(increment_indices, index, side='right')
        items_left -= (pick + 1)
        if pick > 0:
            index -= increment_indices[pick - 1]
        outlist.append(min_pick + pick)
        min_pick += (pick + 1)
    return tuple(outlist)


def scale_series(series, new_range=(0, 1)):
    old_min = min(series)
    old_max = max(series)
    normed = (series - old_min) / (old_max - old_min)
    new_min, new_max = new_range
    new_series = normed * (new_max - new_min) + new_min
    return new_series
