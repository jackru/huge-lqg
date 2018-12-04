"""
Contains wrappers for running and evaluating machine-learning models
on (toy) data.
"""
import time

import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse
from xgboost import XGBRegressor


def model_evaluation(y_test, X_test, fitted_model, training_time, start_time):
    """
    Evaluates model performance and returns a dict of evaluation metrics

    :param 1D array-like y_test: the target variable of the test set
    :param 2D array-like X_test: the predictor variables of the test set
    :param fitted_model: a trained model implementing a predict() method
    :param training_time: the duration of training
    :param start_time: the timestamp at which training started
    :return dict: a dict of evaluation metrics
    """
    return {
        'mse_test': mse(y_test.values, fitted_model.predict(X_test.values)),
        'training_time': training_time,
        'prediction_time': time.time() - start_time - training_time,
    }


def try_scikit_model(X_train, X_test, y_train, y_test, model_kwargs,
                     eval_func=model_evaluation):
    """
    Trains and evaluates a model implementing the scikit-learn API

    :param 2D array-like X_train: the predictor variables of the training set
    :param 2D array-like X_test: the predictor variables of the test set
    :param 1D array-like y_train: the target variable of the training set
    :param 1D array-like y_test: the target variable of the test set
    :param dict model_kwargs: kwargs defining the model to be trained
    :param callable eval_func: function that returns evaluation results
    :return dict: a dict of evaluation metrics
    """
    start = time.time()
    model = model_kwargs['model'](**model_kwargs.get('model_params', {}))
    fitted = model.fit(X_train.values, y_train.values,
                       **model_kwargs.get('fit_params', {}))
    training_time = time.time() - start
    print(
        f"{model_kwargs['model_name']} trained in {training_time:.2f} seconds"
    )
    return eval_func(y_test, X_test, fitted, training_time, start)


def try_xgb_model(X_train, X_test, y_train, y_test, model_kwargs,
                  eval_func=model_evaluation):
    """
    Trains and evaluates a model implementing the scikit-learn API

    :param 2D array-like X_train: the predictor variables of the training set
    :param 2D array-like X_test: the predictor variables of the test set
    :param 1D array-like y_train: the target variable of the training set
    :param 1D array-like y_test: the target variable of the test set
    :param dict model_kwargs: kwargs defining the model to be trained
    :param callable eval_func: function that returns evaluation results
    :return dict: a dict of evaluation metrics
    """
    start = time.time()
    model = XGBRegressor(**model_kwargs.get('model_params', {}))
    if 'early_stopping_rounds' in model_kwargs.get('fit_params', {}):
        model_kwargs['fit_params']['eval_set'] = [
            (X_test.values, y_test.values)
        ]
    fitted = model.fit(X_train.values, y_train.values,
                       **model_kwargs.get('fit_params', {}))
    training_time = time.time() - start
    print(
        f"{model_kwargs['model_name']} trained in {training_time:.2f} seconds"
    )
    return eval_func(y_test, X_test, fitted, training_time, start)


def try_statsmodels_model(X_train, X_test, y_train, y_test, model_kwargs,
                          add_const=True, eval_func=model_evaluation):
    """
    Trains and evaluates a model implementing the statsmodels API

    :param 2D array-like X_train: the predictor variables of the training set
    :param 2D array-like X_test: the predictor variables of the test set
    :param 1D array-like y_train: the target variable of the training set
    :param 1D array-like y_test: the target variable of the test set
    :param bool add_const: whether to allow an intercept
    :param dict model_kwargs: kwargs defining the model to be trained
    :param callable eval_func: function that returns evaluation results
    :return dict: a dict of evaluation metrics
    """
    start = time.time()
    if add_const:
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
    model = model_kwargs['model'](y_train, X_train,
                                  **model_kwargs.get('model_params', {}))
    if model_kwargs.get('regularize', False):
        fitted = model.fit_regularized(**model_kwargs.get('reg_params', {}))
    else:
        fitted = model.fit()
    training_time = time.time() - start
    print(
        f"{model_kwargs['model_name']} trained in {training_time:.2f} seconds"
    )
    return eval_func(y_test, X_test, fitted, training_time, start)


try_map = {
    'scikit': try_scikit_model,
    'statsmodels': try_statsmodels_model,
    'xgb': try_xgb_model,
}


def try_models(X_train, X_test, y_train, y_test,
               param_list, model,
               eval_func=model_evaluation):
    """
    Trains models specified in the lists on the training data, and returns
    evaluations of their performance on the test data

    :param 2D array-like X_train: the predictor variables of the training set
    :param 2D array-like X_test: the predictor variables of the test set
    :param 1D array-like y_train: the target variable of the training set
    :param 1D array-like y_test: the target variable of the test set
    :param list[dict] param_list: each dict specifies a model to try
    :param list[dict] model: the family of models to try
    :param callable eval_func: function that returns evaluation results
    :return dict: evaluation metrics for all models
    """
    results = {}
    for trial in param_list:
        results[f"{trial['model_name']}"] = try_map[model](
            X_train, X_test, y_train, y_test, trial, eval_func=eval_func,
        )
    return results
