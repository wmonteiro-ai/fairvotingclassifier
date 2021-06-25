import numpy as np
import pandas as pd

from Objectives import *

from ConsiderMinimumWeightSumRepair import ConsiderMinimumWeightSumRepair
from FairnessProblem import FairnessProblem

from sklearn.base import ClassifierMixin, MetaEstimatorMixin
from sklearn.metrics import accuracy_score
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.rnsga2 import RNSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from joblib import Parallel, delayed
from topsis import topsis

class FairVotingClassifier(ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, estimators, protected_attribute, unprivileged_values,
                 positive_outcome, all_classes, engine='nsga2',
                 is_protected_ohe=False, privileged_values=None, n_jobs=None,
                 random_state=None, verbose=False):
        self.estimators = estimators
        self.engine = engine
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.is_protected_ohe = is_protected_ohe
        self.positive_outcome = positive_outcome
        self.protected_attribute = protected_attribute
        
        self.unprivileged_values = unprivileged_values
        self.privileged_values = privileged_values
        
        # putting the positive outcome first in the list. It helps when assessing the confusion matrix
        self.all_classes = np.append(positive_outcome, np.delete(np.unique(all_classes), positive_outcome))
        
        self.verbose = verbose
        
    def _choose_weight(self):
        best_index = self._rank_topsis(get_n_best=1).flatten()
        
        self.weights = self.X[best_index]
        self.front_weights = self.F[best_index]
            
    def _collect_proba(self, X_privileged, X_unprivileged, X_unprivileged_counter):
        y_pred_privileged = {}
        for key in X_privileged.keys():
            y_pred_privileged[key] = np.asarray([clf.predict_proba(X_privileged[key]) for clf in self.classifiers])
        
        y_pred_unprivileged = {}
        for key in X_unprivileged.keys():
            y_pred_unprivileged[key] = np.asarray([clf.predict_proba(X_unprivileged[key]) for clf in self.classifiers])
        
        y_pred_unprivileged_counter = {}
        for key in X_unprivileged_counter.keys():
            y_pred_unprivileged_counter[key] = np.asarray([clf.predict_proba(X_unprivileged_counter[key]) for clf in self.classifiers])
        
        return y_pred_privileged, y_pred_unprivileged, y_pred_unprivileged_counter
    
    def _filter_results(self):
        self._remove_duplicates()
        self._pruning()
        self._choose_weight()
    
    def _fit(self, clf, X_train, y_train):
        return clf.fit(X_train, y_train)
        
    def _pruning(self):
        # selecting only the non-dominated solutions (pruning)
        dominated_indexes = []
        for index in range(self.F.shape[0]):
            solution = self.F[index, :]
            if np.sum(np.all(solution <= self.F, axis=1)) > 1:
                dominated_indexes.append(index)

        if self.verbose:
            display(f'{len(dominated_indexes)} dominated out of {self.F.shape[0]}')

        if self.F.shape[0] is not None and self.F.shape[0] > 0:
            self.F[:,2:] = self.F[:,2:] * -1
            
        self.F = np.delete(self.F, dominated_indexes, axis=0)
        self.X = np.delete(self.X, dominated_indexes, axis=0)
        
    def _rank_topsis(self, get_n_best=None, topsis_weights=None):
        if topsis_weights is None:
            topsis_weights = [1/self.problem.num_objectives]*self.problem.num_objectives
        
        # all of them are classified as costs (0) since the Pareto front signal had been
        # inverted for the last three objectives
        topsis_cost = [0]*5
        
        if get_n_best is None and self.F.shape[0] is not None and self.F.shape[0] > 0:
            get_n_best = int(self.F.shape[0]/2)
        
        decision = topsis(self.F, topsis_weights, topsis_cost)
        decision.calc()
        
        best_indexes = np.flip(np.argsort(decision.C)[-get_n_best:])
        return best_indexes
        
    def _remove_duplicates(self):
        _, unique_indexes = np.unique(np.array(self.results.F), return_index=True, axis=0)
        
        # removing duplicates
        if self.verbose:
            display(f'Removing {self.results.F.shape[0] - len(unique_indexes)} duplicates based on the Pareto front approximation.')
        
        self.F = self.results.F[unique_indexes]
        self.X = self.results.X[unique_indexes]
        
    def _validate(self):
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError('The "estimators" attribute must be a list of tuples. ' +
                             'The first attribute is a string with a label and the second ' +
                             'is the estimator itself.'
            )
        names, estimators = zip(*self.estimators)
        return names, estimators
    
    def _predict_proba(self, X, y_true, weights):
        # splitting the dataset between privileged and unprivileged
        Xu, Xp, Xc, yu, yp = split_privileged_unprivileged(X, y_true, self.protected_attribute,
                                                           self.privileged_values, self.unprivileged_values,
                                                           self.is_protected_ohe, self.verbose)

        # getting the forecasts
        yu_pred, yp_pred, yu_counter_pred = self._collect_proba(Xp, Xu, Xc)
        return Xu, Xp, yu, yp, yu_pred, yp_pred, yu_counter_pred
    
    def _predict_proba_weighted(self, yp_pred, yu_pred, yu_counter_pred, weights):
        yp_pred_w = {}
        for key in yp_pred:
            yp_pred_w[key] = [np.average(yp_pred[key], axis=0, weights=weights[len(self.estimators):])]
            
        yu_pred_w = {}
        yu_counter_pred_w = {}
        for key in yu_pred:
            yu_pred_w[key] = [np.average(yu_pred[key], axis=0, weights=weights[:len(self.estimators)])]
            yu_counter_pred_w[key] = [np.average(yu_counter_pred[key], axis=0, weights=weights[len(self.estimators):])]
        
        return yp_pred_w, yu_pred_w, yu_counter_pred_w
    
    def _predict_weighted(self, yp_pred, yu_pred, yu_counter_pred, weights):
        yp_pred_w = {}
        for key in yp_pred:
            yp_pred_w[key] = [np.argmax(np.average(yp_pred[key], axis=0,
                                                   weights=weights[len(self.estimators):]), axis=1)]
            
        yu_pred_w = {}
        yu_counter_pred_w = {}
        for key in yu_pred:
            yu_pred_w[key] = [np.argmax(np.average(yu_pred[key], axis=0,
                                                   weights=weights[:len(self.estimators)]), axis=1)]
            yu_counter_pred_w[key] = [np.argmax(np.average(yu_counter_pred[key], axis=0,
                                                           weights=weights[len(self.estimators):]), axis=1)]
        
        return yp_pred_w, yu_pred_w, yu_counter_pred_w
    
    def _get_results(self, X, y_true, is_proba=False, weights=None):
        if weights is None:
            weights = self.weights.flatten()
            
        Xu, Xp, yu, yp, yp_pred, yu_pred, yu_counter_pred = self._predict_proba(X, y_true, weights)
        
        if not is_proba:
            yp_pred_w, yu_pred_w, yu_counter_pred_w = self._predict_weighted(yp_pred, yu_pred, yu_counter_pred, weights)
        else:
            yp_pred_w, yu_pred_w, yu_counter_pred_w = self._predict_proba_weighted(yp_pred, yu_pred, yu_counter_pred, weights)
        
        y_pred = self._order_results(Xp, Xu, yp_pred_w, yu_pred_w, X.index, is_proba)
        return y_pred, yp, yu, yp_pred_w, yu_pred_w, yu_counter_pred_w
    
    def _order_results(self, Xp, Xu, yp_pred, yu_pred, original_indexes, is_proba=True):
        indexes = np.array([])
        y_pred = np.array([])
        axis = 0 if is_proba else None
        
        for key in Xp:
            indexes = np.append(indexes, Xp[key].index)
            
            if len(y_pred) == 0 and is_proba:
                y_pred = np.asarray(yp_pred[key][0])
            else:
                y_pred = np.append(y_pred, yp_pred[key][0] if is_proba else yp_pred[key], axis=axis)
        
        for key in Xu:
            indexes = np.append(indexes, Xu[key].index)
            
            if len(y_pred) == 0 and is_proba:
                y_pred = np.asarray(yu_pred[key][0])
            else:
                y_pred = np.append(y_pred, yu_pred[key][0] if is_proba else yu_pred[key], axis=axis)
        
        y_pred = pd.DataFrame(y_pred, index=indexes)
        return y_pred.loc[original_indexes].values
        
    def fit(self, X, y, pop_size=1000, n_gen=30, save_history=False,
            sample_weight=None):
        names, self.classifiers = self._validate()

        if self.verbose:
            display(f'Training each classifier.')
            
        # training each classifier
        self.classifiers = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(clf, X, y) for clf in self.classifiers)
        
        if self.verbose:
            [display(f'Classifier {clf} score: {clf.score(X, y)}') for clf in self.classifiers]
        
        # splitting the dataset between privileged and unprivileged
        Xu, Xp, Xc, yu, yp = split_privileged_unprivileged(X, y, self.protected_attribute,
                                                           self.privileged_values, self.unprivileged_values,
                                                           self.is_protected_ohe, self.verbose)
        
        self.X_unprivileged = Xu
        self.X_privileged = Xp
        self.X_unprivileged_counter = Xc
        self.y_true_unprivileged = yu
        self.y_true_privileged = yp
            
        # getting the forecasts
        y_pred_priv, y_pred_unpriv, y_pred_unpriv_counter = self._collect_proba(self.X_privileged, self.X_unprivileged, self.X_unprivileged_counter)
            
        if self.verbose:
            display(f'Starting optimization.')
            
        # optimizing
        ref_points = np.array([[0, 0, -1, -1, -1]])
        if self.engine == 'nsga2':
            algorithm = NSGA2(
                ref_points=ref_points,
                pop_size=pop_size,
                repair=ConsiderMinimumWeightSumRepair(),
                eliminate_duplicates=True
            )
        elif self.engine == 'rnsga2':
            algorithm = RNSGA2(
                ref_points=ref_points,
                pop_size=pop_size,
                repair=ConsiderMinimumWeightSumRepair(),
                eliminate_duplicates=True
            )

        self.problem = FairnessProblem(self.classifiers, self.all_classes, self.y_true_privileged,
                                       self.y_true_unprivileged, y_pred_priv, y_pred_unpriv,
                                       y_pred_unpriv_counter, self.n_jobs)
        
        self.results = minimize(self.problem,
                                algorithm,
                                get_termination('n_gen', n_gen),
                                seed=self.random_state,
                                save_history=save_history,
                                verbose=self.verbose)
        
        if self.verbose:
            display(f'Filtering the results.')
            
        self._filter_results()
        
    def get_pareto(self):
        return self.F, self.X
    
    def predict_proba(self, X, weights=None):
        y_pred, _, _, _, _, _ = self._get_results(X, None, True, weights)
        return y_pred
    
    def predict(self, X, weights=None):
        y_pred, _, _, _, _, _ = self._get_results(X, None, False, weights)
        return y_pred
    
    def rank_solutions(self, get_n_best=None, topsis_weights=None, as_dataframe=False):
        best_indexes = self._rank_topsis(get_n_best=get_n_best, topsis_weights=topsis_weights)
        
        if as_dataframe:
            weight_cols = [f'weight_{i}' for i in range(1, (len(self.estimators)*2)+1)]
            objective_cols = [f'f{i}' for i in range(1, self.problem.num_objectives+1)]
            return pd.DataFrame(np.column_stack([self.X[best_indexes], self.F[best_indexes]]), columns=[weight_cols + objective_cols])
        else:
            return self.F[best_indexes], self.X[best_indexes]
        
    def score(self, X, y_true, weights=None):
        y_pred, yp, yu, yp_pred_w, yu_pred_w, yu_counter_pred_w = self._get_results(X, y_true, False, weights)
        
        # get the Pareto front
        pareto = get_objective_functions(yp, yu, yp_pred_w, yu_pred_w, yu_counter_pred_w,
                                         self.all_classes, self.n_jobs)
        
        # joining both results and returning them in the original order
        return accuracy_score(y_pred, y_true)
    
    def score_pareto(self, X, y_true, weights=None):
        y_pred, yp, yu, yp_pred_w, yu_pred_w, yu_counter_pred_w = self._get_results(X, y_true, False, weights)
        
        # get the Pareto front
        return get_objective_functions(yp, yu, yp_pred_w, yu_pred_w, yu_counter_pred_w, self.all_classes, self.n_jobs)
        