import numpy as np
import warnings

warnings.filterwarnings("ignore")
from scipy.optimize import brute
from typing import Any, Iterable, List, Dict, Tuple
#from statsmodels.tsa.
from ..models.mother import ArimaMother

class SarimaNostra(ArimaMother):

    def __init__(self, serie, max_order_set, train_len, forecast_range
                 , walk_forward=True, alpha=0.05, metric="aic", verbose=True
                 , enforce_complexity=None):
        super().__init__(serie, max_order_set, train_len, forecast_range
                         , walk_forward=walk_forward, alpha=alpha, metric=metric, verbose=verbose)

        pass

    def __fit(self):
        pass

    def __fit_optim_sarima(self):
        pass

    def search_for_the_goodone(self):
        pass
