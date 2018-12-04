"""
Maths functions for toy modelling
"""
import numpy as np
from scipy.special import comb


def binom_row(N, k):
    """
    Returns the number of 'from N choose k' combinations that can be made
    for each possible value of the next pick, where picks are constrained
    to be in ascending order.

    Example:
    Say we want to choose 4 items from [1,2,3,4,5,6]. There are 6C4 = 15
    ways to do this. This function will return the number of combinations
    possible for each initial pick (assuming picks are chosen in ascending
    order). In this case, if we pick 1 first, there are 5C3 = 10 ways to
    pick the remaining 3 numbers. If we pick 2, there are 4C3 = 4 ways.
    Finally, we can pick [3,4,5,6]. The function will return [10,4,1].

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
    :return tuple(int): contains the choices made
    """
    assert nth_comb < comb(from_N, choose_k, exact=True), "Index out of range"
    items_left = from_N
    index = nth_comb
    min_pick = 0
    outlist = []
    for picks_left in range(choose_k, 0, -1):
        increment_indices = (binom_row(items_left, picks_left - 1)
                             .cumsum(dtype=object))
        pick = np.searchsorted(increment_indices, index, side='right')
        items_left -= (pick + 1)
        if pick > 0:
            index -= increment_indices[pick - 1]
        outlist.append(min_pick + pick)
        min_pick += (pick + 1)
    return tuple(outlist)


def scale_series(series, new_range=(0, 1), ci=100):
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
