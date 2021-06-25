import numpy as np
from pymoo.model.repair import Repair

class ConsiderMinimumWeightSumRepair(Repair):
    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            
            if np.all(x == 0):
                x = np.random.random(num_weights)
                pop[k].X = x
        return pop