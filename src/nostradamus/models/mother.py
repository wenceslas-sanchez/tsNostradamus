import numpy as np

from ..utils.error import (exception_type
, check_is_int
, check_is_in
, check_key_is_in
, check_method_lauched)

from typing import Any, Iterable, List, Dict, Tuple


"""
Mother Class for models
"""

class MotherModel():
    """
    Mother class
    """
    def __init__(self):

        pass


class ArimaMother():
    """
    Mother class
    """

    def __init__(self, serie: np.ndarray, max_order_set, train_len: int, forecast_range: int
                 , walk_forward: bool= True, alpha: float= 0.05, metric: str= "aic", verbose: bool= True
                 , enforce_complexity: Iterable[Any]= None):
        """

        :param serie:
        :param max_order_set:
        :param train_len:
        :param forecast_range:
        :param walk_forward:
        :param alpha:
        """

        # Control type
        exception_type(serie, (list, tuple, np.ndarray))
        check_is_int(train_len)
        check_is_int(forecast_range)
        check_is_in(metric.lower(), ["aic", "bic"])

        if enforce_complexity is None:
            # enforce_complexity = {"p": [], "d": [], "q": []}
            enforce_complexity = [0, 0, 0]

        self.serie = serie
        self.max_order_set = max_order_set
        self.len_serie = len(serie)
        self.train_len = train_len
        self.forecast_range = forecast_range
        self.walk_forward = walk_forward
        self.alpha = alpha
        self.metric = metric.lower()
        self.num_forecasting = self.len_serie - self.train_len + 1
        self.verbose = verbose
        self.enforce_complexity = enforce_complexity

        self.fit_ = None
        self.forecast_ = None
        self.method = None
        self.error_by_models = None
        self.error_by_periods = None
        self.metric_error = None

        pass

    def generate_grid_from_param(self) -> Tuple[slice]:
        """
        """
        return tuple([slice(self.enforce_complexity[i], self.max_order_set[i], 1) \
                      for i in range(len(self.max_order_set))])

    def mean_time_weighted(self, mat_pred_during_time: np.ndarray) -> List[np.ndarray]:
        shape_mat_transpose = mat_pred_during_time.T.shape[0]
        stock_wma = []
        num_notzero_col = np.count_nonzero(mat_pred_during_time, axis=0)

        for i in range(shape_mat_transpose):
            test_vec = mat_pred_during_time.T[i]

            vector_pos_nonzero = ~(test_vec == 0) * 1  # recupere

            vector_pos_nonzero = np.where(vector_pos_nonzero != 0, vector_pos_nonzero,
                                          np.nan)  # remplace les 0 pas des nan

            position_nan = ~np.isnan(vector_pos_nonzero)  # je prend la position des valeurs non nan
            cumsum_notnan = np.cumsum(vector_pos_nonzero[position_nan])  # cumsum les valeurs non nan

            vector_pos_nonzero[position_nan] = cumsum_notnan  # integre la cumsum dans le vecteur

            vector_multiplication = np.nan_to_num(vector_pos_nonzero, 0)  # remplace les nan par des 0

            n = num_notzero_col
            divisor = n[i] * (n[i] + 1) / 2

            # weighted average
            wma = np.sum((test_vec * vector_multiplication) / divisor)
            stock_wma.append(wma)

        # Cleaning
        del wma, shape_mat_transpose, num_notzero_col, test_vec \
            , vector_pos_nonzero, position_nan, cumsum_notnan \
            , vector_multiplication, n, divisor, i

        return stock_wma

    def __search_for_the_goodone(self, start):
        return {"1": 1}

    def time_looping(self) -> List[Dict[str, Any]]:
        return [self.__search_for_the_goodone(i) for i in range(self.num_forecasting)]