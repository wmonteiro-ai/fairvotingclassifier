import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from joblib import Parallel, delayed

from sklearn.metrics import accuracy_score

def convert_predictions_2d(y_pred):
    if type(y_pred) is list:
        y_pred = np.array(y_pred)
    
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        y_pred = np.array([y_pred])
    
    return y_pred

def get_privileged_unprivileged_indexes(X, protected_attr, privileged_values, unprivileged_values, 
                                        is_protected_ohe=False, verbose=False):
    privileged_indexes = {}
    unprivileged_indexes = {}
    
    if not is_protected_ohe:
        for value in unprivileged_values:
            entries = X.index[X[protected_attr] == value]
            if len(entries) > 0:
                unprivileged_indexes[value] = entries
            
        for value in privileged_values:
            entries = X.index[X[protected_attr] == value]
            if len(entries) > 0:
                privileged_indexes[value] = entries
    else:
        for value in unprivileged_values:
            entries = np.unique(X[X[value] == 1].index)
            if len(entries) > 0:
                unprivileged_indexes[value] = entries
            
        for value in privileged_values:
            entries = np.unique(X[X[value] == 1].index)
            if len(entries) > 0:
                privileged_indexes[value] = entries
    '''            
    if verbose:
        display(f'Unprivileged index count: ')
        display([f'{key}: {len(unprivileged_indexes[key])}' for key in unprivileged_indexes.keys()])
        
        display(f'Privileged index count: ')
        display([f'{key}: {len(privileged_indexes[key])}' for key in privileged_indexes.keys()])
    '''
    return unprivileged_indexes, privileged_indexes

def split_privileged_unprivileged(X, y, protected_attribute, privileged_values,
                                  unprivileged_values, is_protected_ohe, verbose=False):
    if is_protected_ohe:
        unprivileged_values = [f'{protected_attribute}_{col_name}' for col_name in unprivileged_values]
        privileged_values = [f'{protected_attribute}_{col_name}' for col_name in privileged_values]
    
    # splitting between privileged and unprivileged indexes
    unpriv_indexes, priv_indexes = get_privileged_unprivileged_indexes(X, protected_attribute, privileged_values,
                                                                       unprivileged_values, is_protected_ohe, verbose)
    
    # creating separated tables depending on the indexes 
    X_unprivileged = {}
    X_privileged = {}
    X_unprivileged_counter = {}
    y_true_unpriv = {}
    y_true_priv = {}
    
    for key in unpriv_indexes.keys():
        if y is not None:
            y_true_unpriv[key] = y.loc[unpriv_indexes[key]]
        
        X_unprivileged[key] = X.loc[unpriv_indexes[key]]
        
        # the counterfactual generation takes in account the first privileged value for this attribute
        X_counter = X_unprivileged[key].copy()
        if not is_protected_ohe:
            X_counter[protected_attribute] = list(priv_indexes.keys())[0]
        else:
            X_counter[list(unpriv_indexes.keys())] = 0
            
            try:
                X_counter[list(priv_indexes.keys())[0]] = 1
            except:
                X_counter[privileged_values[0]] = 1
        
        X_unprivileged_counter[key] = X_counter
        
    for key in priv_indexes.keys():
        if y is not None:
            y_true_priv[key] = y.loc[priv_indexes[key]]
    
        X_privileged[key] = X.loc[priv_indexes[key]]
        
    return X_unprivileged, X_privileged, X_unprivileged_counter, y_true_unpriv, y_true_priv
	
def get_algorithm_roc_auc(y_true, y_pred, as_minimization=False, n_jobs=1):
    constant = -1 if as_minimization else 1
    return np.array([roc_auc_score(y_true, ypw, average='weighted')*constant for ypw in convert_predictions_2d(y_pred)])

def get_algorithm_accuracy(y_true, y_pred, as_minimization=False, n_jobs=1):
    constant = -1 if as_minimization else 1
    return np.array([accuracy_score(y_true, ypw)*constant for ypw in convert_predictions_2d(y_pred)])

def get_equalized_odds(y_true_unprivileged, y_true_privileged,
                       y_pred_unprivileged, y_pred_privileged,
                       classes, n_jobs=1):
    y_pred_unprivileged = convert_predictions_2d(y_pred_unprivileged)
    y_pred_privileged = convert_predictions_2d(y_pred_privileged)
    
    unprivileged_rates = np.array(Parallel(n_jobs=n_jobs)(
        delayed(get_fpr_tpr)(y_true_unprivileged, ypw, classes) for ypw in y_pred_unprivileged))
    privileged_rates = np.array(Parallel(n_jobs=n_jobs)(
        delayed(get_fpr_tpr)(y_true_privileged, ypw, classes) for ypw in y_pred_privileged))
    
    fpr_tpr = np.abs(privileged_rates - unprivileged_rates)
    
    return fpr_tpr[:, 0], fpr_tpr[:, 1]

def get_fpr_tpr(y_true, y_pred, classes):
    results = confusion_matrix(y_true, y_pred, labels=classes)
    tpr = results[0, 0]/results.T[0].sum()
    fpr = results[0, 1:].sum()/results[:, 1:].sum()
    
    return np.array([fpr, tpr])

def get_counterfactual_fairness(y_pred_unprivileged, y_pred_unprivileged_counter, as_minimization=False):
    y_pred_unprivileged = convert_predictions_2d(y_pred_unprivileged)
    y_pred_unprivileged_counter = convert_predictions_2d(y_pred_unprivileged_counter)
    
    return np.array([get_algorithm_accuracy(y_pred_unprivileged[i], y_pred_unprivileged_counter[i], as_minimization)
                     for i in range(len(y_pred_unprivileged))])

def get_objective_functions(yp, yu, yp_pred, yu_pred, yu_counter_pred,
                            all_classes, n_jobs, n_weights=1, as_minimization=False):
    '''
    F1 and F2 (Equalized odds): protected and unprotected groups with the same false/true positives.
    F1 is determined as the difference between the FPR (false-positive rate) of the privileged and unprivileged groups.
    F2 is determined as the difference between the TPR (true-positive rate) of the privileged and unprivileged groups.

    The weighted average of the predictions coming from the dataset is taken.
    From it, we select the class (through argmax) and compare this outcome with the real values.
    '''
    y_pred_privileged = []
    y_pred_unprivileged = []
    
    for i in range(n_weights):
        pred = []
        [pred.extend(yp_pred[key][i]) for key in yp_pred.keys()]
        y_pred_privileged.append(np.asarray(pred))

        pred = []
        [pred.extend(yu_pred[key][i]) for key in yu_pred.keys()]
        y_pred_unprivileged.append(np.asarray(pred))
    y_pred_privileged = np.asarray(y_pred_privileged)
    y_pred_unprivileged = np.asarray(y_pred_unprivileged)

    y_true_privileged = []
    y_true_unprivileged = []
    [y_true_privileged.extend(yp[y]) for y in yp]
    [y_true_unprivileged.extend(yu[y]) for y in yu]
    
    f1, f2 = get_equalized_odds(y_true_unprivileged, y_true_privileged,
                                y_pred_unprivileged, y_pred_privileged,
                                all_classes, n_jobs=n_jobs)
    
    '''
    F3 (Counterfactual fairness): same decision if the individual does belong or does not belong to a unprivileged group.
    '''
    y_pred_unprivileged_counter = []
    for i in range(n_weights):
        pred = []
        [pred.extend(yu_counter_pred[key][i]) for key in yu_counter_pred.keys()]
        y_pred_unprivileged_counter.append(np.asarray(pred))
    y_pred_unprivileged_counter = np.asarray(y_pred_unprivileged_counter)

    f3 = get_counterfactual_fairness(y_pred_unprivileged, y_pred_unprivileged_counter, as_minimization)
    
    '''
    F4 and F5 (Algorithm accuracy)
    F4 covers the accuracy for the privileged classes
    F5 covers the accuracy for the unprivileged classes
    '''
    '''
    f4_weights = []
    f4_values = []
    for key in yp.keys():
        f4_weights.append(len(yp[key]))
        f4_values.append(np.array([get_algorithm_roc_auc(yp[key], yp_pred[key][weight])
                   for weight in range(len(yp_pred[key]))]))
    f4 = np.average(f4_values, weights=f4_weights, axis=0)
   
    f5_weights = []
    f5_values = []
    for key in yu.keys():
        f5_weights.append(len(yu[key]))
        f5_values.append(np.array([get_algorithm_roc_auc(yu[key], yu_pred[key][weight])
                   for weight in range(len(yu_pred[key]))]))
    f5 = np.average(f5_values, weights=f5_weights, axis=0)
    '''
    f4 = get_algorithm_roc_auc(y_true_privileged, y_pred_privileged, as_minimization=as_minimization, n_jobs=n_jobs)
    f5 = get_algorithm_roc_auc(y_true_unprivileged, y_pred_unprivileged, as_minimization=as_minimization, n_jobs=n_jobs)
    
    return np.column_stack([f1, f2, f3, f4, f5])
