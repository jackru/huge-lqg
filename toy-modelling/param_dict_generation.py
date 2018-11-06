"""
Contains functions for generating parameter sets.
"""
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
from xgboost import XGBRegressor


sm_family_map = {
    'Poisson': sm.families.Poisson(),
    'Gamma': sm.families.Gamma(),
}

xgb_objective_map = {
    'Poisson': 'count:poisson',
    'Linear': 'reg:linear',
}


def generate_ols_list(alpha_list, L1_wt_list):
    """
    Generates a list of parameter dicts for OLS regression

    :param list[float] alpha_list: a list of alpha values to try
    :param list[float] L1_wt_list: a list of L1 relative weights to try. This
        should be between 0.0 (for ridge/L2 regularization) and 1.0 (for
        lasso/L1 regularization)
    :return list[dict]:
    """
    ols_list = [{
        'model': sm.OLS,
        'model_name': 'OLS_no_reg',
        'regularize': False,
    }]
    for alpha in alpha_list:
        for L1_wt in L1_wt_list:
            ols_list.append({
                'model': sm.OLS,
                'model_name': 'OLS_alpha_' + str(alpha) + '_L1_' + str(L1_wt),
                'regularize': True,
                'reg_params': {
                    'alpha': alpha,
                    'L1_wt': L1_wt,
                },
            })

    return ols_list


def generate_glm_list(family, alpha_list):
    """
    Generates a list of parameter dicts for GLM regression

    :param str family: must be a key in the sm_family_map defined above
    :param list[float] alpha_list: a list of alpha values to try
    :return list[dict]:
    """
    glm_list = [{
        'model': sm.GLM,
        'model_name': 'GLM_' + family + '_no_reg',
        'regularize': False,
        'model_params': {
            'family': sm_family_map[family],
        }
    }]
    for alpha in alpha_list:
        glm_list.append({
            'model': sm.GLM,
            'model_name': 'GLM_' + family + '_alpha_' + str(alpha),
            'regularize': True,
            'model_params': {
                'family': sm_family_map[family],
            },
            'reg_params': {
                'alpha': alpha,
            },
        })

    return glm_list


def generate_mlp_list(hidden_layout_list, max_iter=1000):
    """
    Generates a list of parameter dicts for multi-layer perceptron training

    :param list[tuples] hidden_layout_list: list of hidden layer configurations
    :param int max_iter: the maximum number of learning epochs to train over
    :return list[dict]:
    """
    mlp_list = []
    for layout in hidden_layout_list:
        mlp_list.append({
            'model': MLPRegressor,
            'model_name': 'MLP_' + '_'.join([str(x) for x in layout]),
            'model_params': {
                'hidden_layer_sizes': layout,
                'max_iter': max_iter,
            },
        })
    return mlp_list


def generate_xgb_list(depth_list, learning_rate_list, n_estimators=500,
                      n_jobs=-1, objective='Linear',
                      early_stopping_rounds=None):
    """
    Generates a list of parameter dicts for XGBoost model training

    :param list[int] depth_list: list of tree depths to try
    :param list[float] learning_rate_list: list of learning rates to try
    :param list[float] n_estimators: the maximum number of trees to grow
    :param int n_jobs: the number of cores to use for multiprocessing
    :param str objective: must be a key in the xgb_objective_map defined above
    :param int early_stopping_rounds: training will stop if performance doesn't
        improve for this many consecutive training rounds
    :return list[dict]:
    """
    xgb_list = []
    for depth in depth_list:
        for lr in learning_rate_list:
            xgb_list.append({
                'model': XGBRegressor,
                'model_name': 'XGB_' + str(objective) + '_maxd_' + str(depth)
                              + '_lr_' + str(lr),
                'model_params': {
                    'max_depth': depth,
                    'learning_rate': lr,
                    'n_estimators': n_estimators,
                    'n_jobs': n_jobs,
                    'objective': xgb_objective_map.get(objective, objective),
                },
                'fit_params': {
                    'early_stopping_rounds': early_stopping_rounds,
                    'verbose': False,
                },
            })
    return xgb_list
