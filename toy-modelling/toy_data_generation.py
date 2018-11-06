"""
Contains functions for generating toy data.
"""
import numpy as np
import pandas as pd


def generate_linear_data(nrows, nvars,
                         noise_func=lambda x: (x ** 0.5) * 0.1,
                         const_func=lambda x: np.random.randn() * (x ** 0.5),
                         random_state=None):
    """
    Generates data using a linear generative model

    :param int nrows: the number of data rows output
    :param int nvars: the number of predictor variables
    :param f(nvars) noise_func: determines scale of noise as function of nvars
    :param f(nvars) const_func: determines scale of const as function of nvars
    :param int random_state: specify for reproducable results
    :return tuple(pd.DataFrame, pd.Series): predictors and target variable
    """
    np.random.seed(random_state)
    data = pd.DataFrame(index=range(nrows))
    for varnum in range(nvars):
        data[f'x{varnum}'] = np.random.rand(nrows)
    data['gaussian_noise'] = np.random.randn(nrows) * noise_func(nvars)
    const = const_func(nvars)
    xvars = [x for x in data if x.startswith('x')]
    coefs = pd.Series(np.random.randn(nvars))
    data['y_linear'] = (
        data[xvars].dot(coefs.values)
        + data['gaussian_noise']
        + const
    )
    X = data[xvars]
    y = data['y_linear']
    return (X, y)


def generate_imbalanced_binary_variable(nrows, exponent_of_imbalance=3,
                                        random_state=None):
    """
    Generates a binary random variable with skew towards positive or negative
    class imbalance determined by an exponent

    :param int nrows: the number of binary observations returned
    :param positive, numeric exponent_of_imbalance: determines likelhood of
        variable skewing positive (exponent below 1) or negative (over 1)
    :param int random_state: specify for reproducable results
    :return array of binary-valued floats:
    """
    np.random.seed(random_state)
    threshold = np.random.rand()
    return ((np.random.rand(nrows)**exponent_of_imbalance > threshold)
            .astype(float))


def generate_poisson_data(nrows, nvars, binary_fraction=1.0,
                          binary_imbalance=3, continuous_scaling_factor=0.5,
                          coefs_scaling_factor=0.2, const=np.log(0.01),
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
    :param float coefs_scaling_factor: determines volatility of the response
    :param float const: the base margin, should be entered as np.log(base_rate)
    :param int random_state: specify for reproducable results
    :return tuple(pd.DataFrame, pd.Series): predictors and target variable
    """
    np.random.seed(random_state)
    data = pd.DataFrame(index=range(nrows))
    binvars = int(nvars * binary_fraction)
    for varnum in range(binvars):
        data[f'x{varnum}'] = generate_imbalanced_binary_variable(
            nrows, binary_imbalance)
    for varnum in range(binvars, nvars):
        data[f'x{varnum}'] = np.random.randn(nrows) * continuous_scaling_factor
    coefs = pd.Series(np.random.randn(nvars)) * coefs_scaling_factor
    lam = np.exp(data.dot(coefs.values) + const)
    data['y_poisson'] = np.random.poisson(lam)
    X = data[[x for x in data if x.startswith('x')]]
    y = data['y_poisson']
    return (X, y)


INTERACTION_FUNCS = [
    lambda x, y: abs(x - y),
    lambda x, y: x * y,
    lambda x, y: abs(x * y),
]


def generate_interaction_data(nrows, nvars,
                              interaction_funcs=INTERACTION_FUNCS,
                              noise_func=lambda x: (x ** 0.5) * 0.1,
                              const_func=(
                                  lambda x: np.random.randn() * (x ** 0.5)
                              ),
                              random_state=None):
    """
    Generates data using a linear generative model, with two-way interactions
    between variables.

    :param int nrows: the number of data rows output
    :param int nvars: the number of predictor variables
    :param list[functions] interaction_funcs: a list of functions specifying
        variable interactions
    :param f(nvars) noise_func: determines scale of noise as function of nvars
    :param f(nvars) const_func: determines scale of const as function of nvars
    :param int random_state: specify for reproducable results
    :return tuple(pd.DataFrame, pd.Series): predictors and target variable
    """
    np.random.seed(random_state)
    data = pd.DataFrame(index=range(nrows))
    for varnum in range(nvars):
        data[f'x{varnum}'] = np.random.randn(nrows)
    data['gaussian_noise'] = np.random.randn(nrows) * noise_func(nvars)
    const = const_func(nvars)
    xvars = [x for x in data if x.startswith('x')]
    coefs = pd.Series(np.random.randn(nvars))
    data['y_interactions'] = (
        data[xvars].dot(coefs.values)
        + data['gaussian_noise']
        + const
    )
    for i in range(nvars - 1):
        for j in range(i+1, nvars):
            func = np.random.choice(interaction_funcs)
            coef = np.random.randn()
            data['y_interactions'] += (
                coef * func(data[f'x{i}'], data[f'x{j}'])
            )

    X = data[xvars]
    y = data['y_interactions']

    return (X, y)
