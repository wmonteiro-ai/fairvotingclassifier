import numpy as np
import pandas as pd

from pymoo.model.problem import Problem
from joblib import Parallel, delayed
from Objectives import *

def get_weight_sum(weights):
    return -np.sum(weights, axis=1) + 1

class FairnessProblem(Problem):
    def __init__(self, estimators, classes, 
                 y_true_privileged, y_true_unprivileged,
                 y_pred_privileged, y_pred_unprivileged, y_pred_unprivileged_counter, n_jobs=1):
        self.estimators = estimators
        self.num_weights = len(estimators)*2
        self.num_objectives = 5
        self.n_jobs = n_jobs
        
        self.classes = classes
        
        self.y_true_privileged = y_true_privileged
        self.y_true_unprivileged = y_true_unprivileged
        
        self.y_pred_privileged = y_pred_privileged
        self.y_pred_unprivileged = y_pred_unprivileged
        self.y_pred_unprivileged_counter = y_pred_unprivileged_counter
            
        super().__init__(n_var=self.num_weights,
                         n_obj=self.num_objectives,
                         n_constr=2,
                         xl=np.array([0]*self.num_weights),
                         xu=np.array([100]*self.num_weights))
    
    def _evaluate(self, X, out, *args, **kwargs):
        #F1: equalized odds (FPR)
        #F2: equalized odds (TPR)
        #F3: counterfactual fairness
        #F4: accuracy (privileged)
        #F5: accuracy (unprivileged)
        
        #G1, G2: the sums of all the weights must be equal to or greater than 1
        #arXiv 2001.09784.pdf
        #arXiv 1812.06135.pdf
        #arXiv 1908.09635.pdf
        
        yp_pred = {}
        for key in self.y_pred_privileged:
            yp_pred[key] = [np.argmax(np.average(self.y_pred_privileged[key],
                                                 axis=0, weights=weight[len(self.estimators):]), axis=1)
                            for weight in X]
            
        yu_pred = {}
        yu_counter_pred = {}
        for key in self.y_pred_unprivileged:
            yu_pred[key] = [np.argmax(np.average(self.y_pred_unprivileged[key],
                                                 axis=0, weights=weight[:len(self.estimators)]), axis=1)
                            for weight in X]
            yu_counter_pred[key] = [np.argmax(np.average(self.y_pred_unprivileged_counter[key],
                                                         axis=0, weights=weight[len(self.estimators):]), axis=1)
                                    for weight in X]
        
        out["F"] = get_objective_functions(self.y_true_privileged, self.y_true_unprivileged,
                                           yp_pred, yu_pred, yu_counter_pred, self.classes, 
                                           self.n_jobs, n_weights=len(X), as_minimization=True)
        
        # Constraints
        g1 = get_weight_sum(X[:, len(self.estimators):])
        g2 = get_weight_sum(X[:, :len(self.estimators)])
        out["G"] = np.column_stack([g1, g2])