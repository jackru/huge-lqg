import numpy as np

def gini(y_true, y_pred, exposure=None):
    """Calculates absolute and normalised gini coefficents."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if exposure is None:
        exposure = np.ones(len(y_true))
    else:
        exposure = np.array(exposure)

    assert len(y_true) == len(y_pred) == len(exposure)
    
    ranks_pred = (y_pred / exposure).argsort()
    exposure_pred_sort = exposure[ranks_pred]
    response_pred_sort = y_true[ranks_pred]
    exposure_pred_norm = exposure_pred_sort.cumsum() / exposure_pred_sort.sum()
    response_pred_norm = response_pred_sort.cumsum() / response_pred_sort.sum()
    
    ranks_true = (y_true / exposure).argsort()
    exposure_true_sort = exposure[ranks_true]
    response_true_sort = y_true[ranks_true]
    exposure_true_norm = exposure_true_sort.cumsum() / exposure_true_sort.sum()
    response_true_norm = response_true_sort.cumsum() / response_true_sort.sum()
    
    auc_pred = 0.5 - np.trapz(response_pred_norm, exposure_pred_norm)
    auc_true = 0.5 - np.trapz(response_true_norm, exposure_true_norm)
    
    gini_abs = auc_pred * 2
    gini_norm = auc_pred / auc_true
    
    return gini_abs, gini_norm
