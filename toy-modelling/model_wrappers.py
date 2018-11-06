"""
Contains wrappers for running and evaluating machine-learning models
on (toy) data.
"""
import time

import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse
from xgboost import XGBRegressor


def model_evaluation(y_test, X_test, fitted_model, training_time, start_time):
    return {
        'mse_test': mse(y_test.values, fitted_model.predict(X_test.values)),
        'training_time': training_time,
        'prediction_time': time.time() - training_time - start_time,
    }


def try_scikit_model(X_train, X_test, y_train, y_test, model_kwargs,
                     eval_func=model_evaluation):
    start = time.time()
    model = model_kwargs['model'](**model_kwargs.get('model_params', {}))
    if ((model_kwargs['model'] == XGBRegressor)
            & ('early_stopping_rounds' in model_kwargs.get('fit_params', {}))):
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
                          eval_func=model_evaluation):
    start = time.time()
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    model = model_kwargs['model'](y_train, X_train_const,
                                  **model_kwargs.get('model_params', {}))
    if model_kwargs.get('regularize', False):
        fitted = model.fit_regularized(**model_kwargs.get('reg_params', {}))
    else:
        fitted = model.fit()
    training_time = time.time() - start
    print(
        f"{model_kwargs['model_name']} trained in {training_time:.2f} seconds"
    )
    return eval_func(y_test, X_test_const, fitted, training_time, start)
