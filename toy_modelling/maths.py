"""
Maths functions for toy modelling
"""
import numpy as np
from scipy.special import comb


def binom_row(N, k):
    """
    Returns the number of 'from N choose k' combinations that can be made
    for each possible value of the next pick.

    :param int N: the number of options left
    :param int k: the number of picks left
    :return list[int]: the number of combinations that can be made with
        each possible value (in ascending order) of the next pick
    """
    outlist = []
    for i in range(k, N):
        outlist.insert(0, comb(i, k, exact=True))
    return np.array(outlist, dtype=object)


def nth_combination(from_N, choose_k, nth_comb):
    """
    Returns the nth 'from N choose k' combination that can be made
    where these combinations are sorted in ascending order.

    :param int from_N: the number of options to choose from
    :param int choose_k: the number of distinct choices to make
    :param nth_comb: the index of the required combination from the
        ascending-sorted set of combinations
    return tuple(int): contains the choices made
    """
    assert nth_comb < comb(from_N, choose_k, exact=True), "Index out of range"
    items_left = from_N
    index = nth_comb
    min_pick = 0
    outlist = []
    for picks_left in range(choose_k, 0, -1):
        increment_indices = binom_row(items_left, picks_left - 1).cumsum(dtype=object)
        pick = np.searchsorted(increment_indices, index, side='right')
        items_left -= (pick + 1)
        if pick > 0:
            index -= increment_indices[pick - 1]
        outlist.append(min_pick + pick)
        min_pick += (pick + 1)
    return tuple(outlist)


def scale_series(series, ci=100, new_range=(0, 1)):
    """
    Scales a series by translation and multiplication, preserving the
    relative position of all items in the series.

    :param array-like series: the series to be scaled
    :param numeric ci: the percentage of data that will fall within the new
        range (tails either side of this range will have equal weights)
    :param tuple(numeric) new_range: the new bounds of the quantile range
        specified by `ci` after scaling
    :return array-like series: the scaled series
    """
    lower_q = (100 - ci) / 2
    upper_q = (100 + ci) / 2
    old_lower_q = np.percentile(series, lower_q)
    old_upper_q = np.percentile(series, upper_q)
    normed = (series - old_lower_q) / (old_upper_q - old_lower_q)
    new_lower_q, new_upper_q = new_range
    new_series = normed * (new_upper_q - new_lower_q) + new_lower_q
    return new_series
