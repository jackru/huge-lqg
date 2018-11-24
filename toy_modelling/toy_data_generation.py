"""
Contains functions for generating toy data.
"""
from random import sample, seed

import numpy as np
import pandas as pd
from scipy.special import comb

from maths import nth_combination, scale_series


def get_rng(random_state=None):
    """Returns a RandomState object given None, int or RandomState input"""
    return (
        random_state if type(random_state) is np.random.mtrand.RandomState
        else np.random.RandomState(random_state)
    )

def generate_binary_var(nrows, exponent_of_imbalance=3, random_state=None):
    """
    Generates a binary random variable with skew towards positive or negative
    class imbalance determined by an exponent

    :param int nrows: the number of binary observations returned
    :param positive, numeric exponent_of_imbalance: determines likelhood of
        variable skewing positive (exponent below 1) or negative (over 1)
    :param int random_state: specify for reproducible results
    :return array of binary-valued floats:
    """
    rng = get_rng(random_state)
    threshold = rng.rand()
    return ((rng.rand(nrows)**exponent_of_imbalance > threshold)
            .astype(float))


def generate_x_data(nrows, nvars, binary_fraction=1.0, binary_imbalance=3,
                    continuous_scaling_factor=0.5, random_state=None):
    """
    Generates basic predictor data.

    :param int nrows: the number of data rows output
    :param int nvars: the number of predictor variables
    :param float binary_fraction: the fraction of generated variables that are
        binary (the remainder will be continuous normally distributed)
    :param positive, numeric binary_imbalance: determines likelhood of binary
        variables skewing positive (exponent below 1) or negative (over 1)
    :param float continuous_scaling_factor: scale of continuous variable value
        range relative to binary variables
    :param int random_state: specify for reproducible results
    :return pd.DataFrame: the data
    """
    rng = get_rng(random_state)
    data = pd.DataFrame(index=range(nrows))
    binvars = int(nvars * binary_fraction)
    for varnum in range(binvars):
        data[f'x{varnum}'] = generate_binary_var(nrows, binary_imbalance, rng)
    for varnum in range(binvars, nvars):
        data[f'x{varnum}'] = rng.randn(nrows) * continuous_scaling_factor
    return data


ARRAY_LIST_FUNCS = {
    'product': lambda array_list: np.product(array_list, axis=0),
    'max': lambda array_list: np.max(array_list, axis=0),
    'min': lambda array_list: np.min(array_list, axis=0),
    'range': lambda array_list: (np.max(array_list, axis=0)
                                 - np.min(array_list, axis=0)),
}


def generate_systematic_y(x_data, terms=[(1, 1.0)],
                          interaction_funcs=ARRAY_LIST_FUNCS,
                          debug=False, scale_to_range=None, random_state=None):
    """
    Generates contributing terms and multiplies each by a coefficient.

    :param pd.DataFrame x_data: the predictor variables
    :param list[tuple(int, numeric)] terms: determines the number of terms of
        each order that will contribute to y
    :param list[functions] interaction_funcs: a list of functions specifying
        variable interactions
    :param bool debug: if True, returns information on the generative model
    :param tuple(min, max) scale_to_range: if specified, the response will be
        scaled to this range
    :param int random_state: specify for reproducible results
    :return pd.Series: the response variable
    :return dict: (optional) the debug info
    """
    rng = get_rng(random_state)
    seed(rng)
    nrows, nvars = x_data.shape
    if debug:
        debug_dict = {}
    y = np.zeros(nrows)
    for order, term_count in terms:
        n_combs = comb(nvars, order, exact=True)
        if type(term_count) == float:
            term_count = int(n_combs * term_count)
        choices = sample(range(n_combs), term_count)
        for i in sorted(choices):
            if order == 1:
                combination = i
                func = 'identity'
                values = x_data[f'x{i}']
            else:
                combination = nth_combination(nvars, order, i)
                data_col_list = [x_data[f'x{j}'] for j in combination]
                func = rng.choice(list(interaction_funcs))
                values = interaction_funcs[func](data_col_list)
            coef = rng.randn()
            if debug:
                debug_dict[combination] = {'func': func, 'coef': coef}
            y += coef * values
    if scale_to_range:
        y = scale_series(y, scale_to_range)
    if debug:
        return y, debug_dict
    else:
        return y


def generate_linear_data(nrows, nvars, binary_fraction=1.0, binary_imbalance=3,
                         continuous_scaling_factor=0.5, noise_scalar=1,
                         terms=[(1, 1.0)], interaction_funcs=ARRAY_LIST_FUNCS,
                         scale_to_range=None, random_state=None):
    """
    Generates data using a linear generative model

    :param int nrows: the number of data rows output
    :param int nvars: the number of predictor variables
    :param float binary_fraction: the fraction of generated variables that are
        binary (the remainder will be continuous normally distributed)
    :param positive, numeric binary_imbalance: determines likelhood of binary
        variables skewing positive (exponent below 1) or negative (over 1)
    :param float continuous_scaling_factor: scale of continuous variable value
        range relative to binary variables
    :param noise_scalar numeric: determines the scale of noise added to the
        systematic component of the response
    :param list[tuple(int, numeric)] terms: determines the number of terms of
        each order that will contribute to y
    :param list[functions] interaction_funcs: a list of functions specifying
        variable interactions
    :param tuple(min, max) scale_to_range: if specified, the response will be
        scaled to this range
    :param int random_state: specify for reproducible results
    :return tuple(pd.DataFrame, pd.Series): predictors and target variable
    """
    rng = get_rng(random_state)
    data = generate_x_data(nrows, nvars, binary_fraction, binary_imbalance,
                           continuous_scaling_factor, rng)
    data['y_linear'] = (
        generate_systematic_y(
            data, terms=terms, interaction_funcs=interaction_funcs,
            scale_to_range=scale_to_range, random_state=rng
        )
        + np.random.randn(nrows) * noise_scalar
    )
    X = data[[x for x in data if x.startswith('x')]]
    y = data['y_linear']
    return (X, y)


def generate_poisson_data(nrows, nvars, binary_fraction=1.0,
                          binary_imbalance=3, continuous_scaling_factor=0.5,
                          terms=[(1, 1.0)], interaction_funcs=ARRAY_LIST_FUNCS,
                          scale_to_range=(0.0003, 0.3),
                          random_state=None):
    """
    Generates data using a Poisson generative model

    :param int nrows: the number of data rows output
    :param int nvars: the number of predictor variables
    :param float binary_fraction: fraction of variables that are binary-valued
    :param float binary_imbalance: determines likelhood of binary variables
        skewing positive (below 1) or negative (over 1)
    :param float continuous_scaling_factor: scale of continuous variable value
        range relative to binary variables
    :param list[tuple(int, numeric)] terms: determines the number of terms of
        each order that will contribute to y
    :param list[functions] interaction_funcs: a list of functions specifying
        variable interactions
    :param tuple(min, max) scale_to_range: if specified, the lambda used to
        generate samples will be scaled to this range
    :param int random_state: specify for reproducible results
    :return tuple(pd.DataFrame, pd.Series): predictors and target variable
    """
    rng = get_rng(random_state)
    data = generate_x_data(nrows, nvars, binary_fraction, binary_imbalance,
                           continuous_scaling_factor, rng)
    if scale_to_range:
        scale_to_range = tuple(np.log(scale_to_range))
    lam = np.exp(generate_systematic_y(
              data, terms=terms, interaction_funcs=interaction_funcs,
              scale_to_range=scale_to_range, random_state=rng
          ))
    data['y_poisson'] = rng.poisson(lam)
    X = data[[x for x in data if x.startswith('x')]]
    y = data['y_poisson']
    return (X, y)


def generate_gamma_data(nrows, nvars, binary_fraction=1.0,
                        binary_imbalance=3, continuous_scaling_factor=0.5,
                        terms=[(1, 1.0)], interaction_funcs=ARRAY_LIST_FUNCS,
                        scale_to_range=(100, 1000000),
                        shape=1, random_state=None):
    """
    Generates data using a Poisson generative model

    :param int nrows: the number of data rows output
    :param int nvars: the number of predictor variables
    :param float binary_fraction: fraction of variables that are binary-valued
    :param float binary_imbalance: determines likelhood of binary variables
        skewing positive (below 1) or negative (over 1)
    :param float continuous_scaling_factor: scale of continuous variable value
        range relative to binary variables
    :param list[tuple(int, numeric)] terms: determines the number of terms of
        each order that will contribute to y
    :param list[functions] interaction_funcs: a list of functions specifying
        variable interactions
    :param tuple(min, max) scale_to_range: if specified, the mu used to
        generate samples will be scaled to this range
    :param int shape: the gamma distribution shape parameter
    :param int random_state: specify for reproducible results
    :return tuple(pd.DataFrame, pd.Series): predictors and target variable
    """
    rng = get_rng(random_state)
    data = generate_x_data(nrows, nvars, binary_fraction, binary_imbalance,
                           continuous_scaling_factor, rng)
    if scale_to_range:
        scale_to_range = tuple(np.log(scale_to_range))
    mu = np.exp(generate_systematic_y(
              data, terms=terms, interaction_funcs=interaction_funcs,
              scale_to_range=scale_to_range, random_state=rng
          ))
    scale = mu / shape
    data['y_gamma'] = rng.gamma(shape, scale)
    X = data[[x for x in data if x.startswith('x')]]
    y = data['y_gamma']
    return (X, y)
