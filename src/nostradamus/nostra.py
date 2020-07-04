import numpy as np
from statsmodels.tsa.arima_model import ARIMA

class BasicNostra():

    def __init__(self):
        print("Classe nostra BB")
        pass

    def fit(self, serie, window, params, walk_forward= True):
        pass

    def all_3_permutations(self, params):
        """

        Input :
            - perm_list : tuple or list like ;

        Output : list like : each possible permutations from perm_list parameters

        """

        return [[i, j, k]
                for i in range(params[0] + 1)
                for j in range(params[1] + 1)
                for k in range(params[2] + 1)]
