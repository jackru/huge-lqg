"""
Contains functions for generating toy data.
"""
import numpy as np
import pandas as pd


def generate_linear_data(nrows, nvars,
                         noise_func=lambda x: (x ** 0.5) * 0.1,
                         const_func=lambda x: np.random.randn() * (x ** 0.5),
                         random_state=None):
    np.random.seed(random_state)
    data = pd.DataFrame(index=range(nrows))
    for varnum in range(nvars):
        data[f'x{varnum}'] = np.random.rand(nrows)
    data['gaussian_noise'] = np.random.randn(nrows) * noise_func(nvars)
    const = const_func(nvars)
    xvars = [x for x in data if x.startswith('x')]
    coefs = pd.Series(np.random.randn(nvars))
    data['y_linear'] = data[xvars].dot(coefs.values) + data['gaussian_noise'] + const
    X = data[xvars]
    y = data['y_linear']
    return (X, y)


def generate_imbalanced_binary_variable(nrows, exponent_of_imbalance=3, random_state=None):
    np.random.seed(random_state)
    threshold = np.random.rand()
    return (np.random.rand(nrows)**exponent_of_imbalance > threshold).astype(float)


def generate_poisson_data(nrows, nvars, binary_fraction=1.0, binary_imbalance=3,
                          continuous_scaling_factor=0.5, coefs_scaling_factor=0.2,
                          const=np.log(0.01), random_state=None):
    np.random.seed(random_state)
    data = pd.DataFrame(index=range(nrows))
    binvars = int(nvars * binary_fraction)
    for varnum in range(binvars):
        data[f'x{varnum}'] = generate_imbalanced_binary_variable(nrows, binary_imbalance)
    for varnum in range(binvars, nvars):
        data[f'x{varnum}'] = np.random.randn(nrows) * continuous_scaling_factor
    coefs = pd.Series(np.random.randn(nvars)) * coefs_scaling_factor
    lam=np.exp(data.dot(coefs.values) + const)
    data['y_poisson'] = np.random.poisson(lam)
    X = data[[x for x in data if x.startswith('x')]]
    y = data['y_poisson']
    return (X, y)


def generate_interaction_data(nrows, nvars,
                              noise_func=lambda x: (x ** 0.5) * 0.1,
                              const_func=lambda x: np.random.randn() * (x ** 0.5),
                              random_state=None):
    np.random.seed(random_state)
    data = pd.DataFrame(index=range(nrows))
    for varnum in range(nvars):
        data[f'x{varnum}'] = np.random.rand(nrows)
    
    for i in range(nvars - 1):
        for j in range(i+1, nvars):
            func = np.random.choice([
                lambda x, y: x - y,
                lambda x, y: abs(x - y),
                lambda x, y: x * y,
                lambda x, y: abs(x * y),
            ])
            data[f'i{i}j{j}'] = func(data[f'x{i}'], data[f'x{j}'])
    
    data['gaussian_noise'] = np.random.randn(nrows) * noise_func(nvars)
    const = const_func(nvars)
    xvars = [x for x in data if x.startswith('x')]
    interactions = [ij for ij in data if ij.startswith('i')]
    coefs = pd.Series(np.random.randn(nvars))
    coefs_interaction = pd.Series(np.random.randn(int(nvars * (nvars - 1) / 2)))
    data['y_interactions'] = (
        data[xvars].dot(coefs.values)
        + data[interactions].dot(coefs_interaction.values)
        + data['gaussian_noise']
        + const
    )
    X = data[xvars]
    y = data['y_interactions']
    return (X, y)
