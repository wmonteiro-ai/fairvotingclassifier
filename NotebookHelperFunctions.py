from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from aif360.sklearn.inprocessing import AdversarialDebiasing
from aif360.sklearn.postprocessing import *
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import *
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd

from Objectives import *

def get_classifiers(seed, max_iter):
    # As recommended by Watson AutoAI
    clf1 = LGBMClassifier(random_state=seed)
    clf2 = GradientBoostingClassifier(random_state=seed)

    # As recommended by Azure AutoML
    clf3 = LogisticRegression(random_state=seed, max_iter=max_iter)
    clf4 = SGDClassifier(loss='log', random_state=seed)

    return [('lgb', clf1), ('gb', clf2), ('lr', clf3), ('sgd', clf4)]

def get_predictions_general(clf, cat_columns_dummies, protected_attr, X, X_unprivileged, X_privileged, X_unprivileged_counter,
                            sensitive_features=None, random_state=None, is_aif=False):
    if sensitive_features is not None:
        y_pred = clf.predict(X, sensitive_features=sensitive_features, random_state=random_state)
        y_pred_privileged = dict((X, np.array([clf.predict(X_privileged[X], sensitive_features=sensitive_features.loc[X_privileged[X].index], random_state=random_state)])) for X in X_privileged)
        y_pred_unprivileged = dict((X, np.array([clf.predict(X_unprivileged[X], sensitive_features=sensitive_features.loc[X_unprivileged[X].index], random_state=random_state)])) for X in X_unprivileged)
        y_pred_unprivileged_counter = dict((X, np.array([clf.predict(X_unprivileged_counter[X], sensitive_features=sensitive_features.loc[X_unprivileged_counter[X].index], random_state=random_state)])) for X in X_unprivileged_counter)
    elif is_aif:
        y_pred = clf.predict(convert_df_to_aif_inprocessing(X, cat_columns_dummies, protected_attr))
        y_pred_privileged = dict((X, np.array([clf.predict(convert_df_to_aif_inprocessing(X_privileged[X], cat_columns_dummies, protected_attr))])) for X in X_privileged)
        y_pred_unprivileged = dict((X, np.array([clf.predict(convert_df_to_aif_inprocessing(X_unprivileged[X], cat_columns_dummies, protected_attr))])) for X in X_unprivileged)
        y_pred_unprivileged_counter = dict((X, np.array([clf.predict(convert_df_to_aif_inprocessing(X_unprivileged_counter[X], cat_columns_dummies, protected_attr))])) for X in X_unprivileged_counter)
    else:
        y_pred = clf.predict(X)
        y_pred_privileged = dict((X, np.array([clf.predict(X_privileged[X])])) for X in X_privileged)
        y_pred_unprivileged = dict((X, np.array([clf.predict(X_unprivileged[X])])) for X in X_unprivileged)
        y_pred_unprivileged_counter = dict((X, np.array([clf.predict(X_unprivileged_counter[X])])) for X in X_unprivileged_counter)

    return y_pred, y_pred_privileged, y_pred_unprivileged, y_pred_unprivileged_counter

def get_predictions_weighted(clf, X, X_unprivileged, X_privileged, X_unprivileged_counter, weights=None):
    y_pred = clf.predict(X, weights=weights)
    y_pred_privileged = dict((X, np.array([clf.predict(X_privileged[X], weights=weights)])) for X in X_privileged)
    y_pred_unprivileged = dict((X, np.array([clf.predict(X_unprivileged[X], weights=weights)])) for X in X_unprivileged)
    y_pred_unprivileged_counter = dict((X, np.array([clf.predict(X_unprivileged_counter[X], weights=weights)])) for X in X_unprivileged_counter)

    return y_pred, y_pred_privileged, y_pred_unprivileged, y_pred_unprivileged_counter

# remove random_state
def score_classifier_general(clf, cat_columns_dummies, X, y_true, protected_attr, privileged_values, unprivileged_values,
                             is_protected_ohe, all_classes, best_outcome, sensitive_features=None,
                             verbose=False, n_jobs=1, random_state=None, is_aif=False, is_protected_index=False,
                             weights=None):
    if is_protected_index:
        X = X.reset_index()
    
    # standardazing the output to pandas
    y_true = pd.Series(y_true)
    y_true.index = X.index

    Xu, Xp, Xc, yu, yp = split_privileged_unprivileged(X, y_true, protected_attr, privileged_values,
                                                       unprivileged_values, is_protected_ohe, verbose)
    
    if is_protected_index:
        X = X.set_index(protected_attr)
        y_true.index = X.index
        
        for key in Xp.keys():
            Xp[key] = Xp[key].set_index(protected_attr)
            yp[key].index = Xp[key].index
            
        for key in Xu.keys():
            Xu[key] = Xu[key].set_index(protected_attr)
            yu[key].index = Xu[key].index
            
        for key in Xc.keys():
            Xc[key] = Xc[key].set_index(protected_attr)
    
    # getting the predictions
    if weights is None:
        y_pred, yp_pred, yu_pred, yu_counter_pred = get_predictions_general(clf, cat_columns_dummies, protected_attr, X, Xu, Xp, Xc, sensitive_features,
                                                                            random_state, is_aif)
    else:
        y_pred, yp_pred, yu_pred, yu_counter_pred = get_predictions_weighted(clf, X, Xu, Xp, Xc, weights)
    
    return get_objective_functions(yp, yu, yp_pred, yu_pred, yu_counter_pred,
                                   all_classes, n_jobs, as_minimization=False)
								   

# Original classifiers, scikit-lego and voting classifier
def get_standard_classifier_results(classifier, cat_columns_dummies, X_test, y_test, technique_name, algorithm, all_classes,
                                    best_outcome, is_protected_ohe, n_jobs, protected_attr, privileged_values, unprivileged_values,
									seed, verbose=False, is_aif=False, weights=None):
    run_info = [[technique_name, algorithm, type(classifier).__name__, seed]]
    results = score_classifier_general(classifier, cat_columns_dummies, X_test, y_test, protected_attr, privileged_values,
                                       unprivileged_values, is_protected_ohe, all_classes,
                                       best_outcome, verbose=verbose, n_jobs=n_jobs, is_aif=is_aif, weights=weights)
    
    return results, run_info

# Fairlearn
def get_fairlearn_results(classifier, cat_columns_dummies, X_train, y_train, X_test, y_test, X_protected_train, X_protected_test, all_classes,
						  best_outcome, is_protected_ohe, n_jobs, protected_attr, privileged_values, unprivileged_values, seed, verbose=False):
    results = []
    run_info = []
    constraints = {'F1': 'false_positive_rate_parity', 'F2': 'true_positive_rate_parity', 'F1_F2': 'equalized_odds'}
    
    # Fairlearn
    for constraint in constraints:
        mitigator = ThresholdOptimizer(estimator=classifier,
                                       constraints=constraints[constraint],
                                       objective='balanced_accuracy_score')
        
        mitigator.fit(X_train, y_train, sensitive_features=X_protected_train)
        
        run_info.extend([['Fairlearn', constraints[constraint], type(classifier).__name__, seed]])
        
        results.extend(score_classifier_general(mitigator, cat_columns_dummies, X_test, y_test, protected_attr, privileged_values,
                                                unprivileged_values, is_protected_ohe, all_classes, best_outcome,
                                                sensitive_features=X_protected_test, n_jobs=n_jobs, verbose=False, random_state=seed))
    
    return results, run_info

# AIF360
def convert_df_to_aif_inprocessing(X, cat_columns_dummies, protected_attr):
    protected_columns = [col for col in cat_columns_dummies if col.startswith(protected_attr)]
    X_index = X[protected_columns].idxmax(axis=1)
    X_index.name = protected_attr
    
    return X.set_index(X_index).drop(protected_columns, axis=1)

def convert_df_to_aif_postprocessing(X, cat_columns_dummies, protected_attr, privileged_values, unprivileged_values):
    protected_columns = [col for col in cat_columns_dummies if col.startswith(protected_attr)]
    
    X_index = X[protected_columns].idxmax(axis=1)
    for privileged in privileged_values:
        X_index[X_index==f'{protected_attr}_{privileged}'] = 1
    for unprivileged in unprivileged_values:
        X_index[X_index==f'{protected_attr}_{unprivileged}'] = 0
    X_index.name = protected_attr
    
    return X.set_index(X_index).drop(protected_columns, axis=1)

def get_aif_postprocessing_classifier_results(classifier, cat_columns_dummies, protected_attr, X_test, y_test, technique_name, algorithm, all_classes,
                                              best_outcome, is_protected_ohe, n_jobs, seed, verbose=False, weights=None):
    run_info = [[technique_name, algorithm, type(classifier).__name__, seed]]
    results = score_classifier_general(classifier, cat_columns_dummies, X_test, y_test, protected_attr, [1],
                                       [0], is_protected_ohe, all_classes, best_outcome, verbose=verbose,
                                       n_jobs=n_jobs, is_aif=False, weights=weights, is_protected_index=True)
    
    return results, run_info