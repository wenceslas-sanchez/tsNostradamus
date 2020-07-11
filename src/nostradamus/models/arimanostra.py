import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima_model import ARIMA

from ..utils.error import (exception_type
                            , check_is_int
                            , check_is_in
                            , check_key_is_in
                           )

from .mother import Model


class ArimaNostra(Model):

    def __init__(self, serie, max_order_set, train_len, forecast_range
                 , walk_forward= True, alpha= 0.05, metric= "aic", verbose= True
                 , enforce_complexity=None):
        """

        :param serie:
        :param max_order_set:
        :param train_len:
        :param forecast_range:
        :param walk_forward:
        :param alpha:
        """
        super().__init__()

        if enforce_complexity is None:
            enforce_complexity = {"p": [0], "d": [0], "q": [0]}

        self.serie= serie
        self.max_order_set= max_order_set
        self.len_serie= len(serie)
        self.train_len= train_len
        self.forecast_range= forecast_range
        self.walk_forward= walk_forward
        self.alpha= alpha
        self.metric= metric
        self.num_forecasting= self.len_serie - self.train_len + 1
        self.verbose= verbose
        self.enforce_complexity= enforce_complexity


        # Control type
        exception_type(self.serie, (list, tuple, np.ndarray))
        check_is_int(self.train_len)
        check_is_int(self.forecast_range)

        check_is_in(self.metric, ["aic", "bic"])
        check_key_is_in(["p", "d", "q"], self.enforce_complexity)
        pass

    def __all_3_permutations(self):
        """

        :param max_order_set:
        :return: each possible permutations from perm_list parameters
        """
        return [[i, j, k]
                for i in range(self.max_order_set[0] + 1)
                for j in range(self.max_order_set[1] + 1)
                for k in range(self.max_order_set[2] + 1)]


    def __fit(self, start, order):
        """

        :param start:
        :param order:
        :return:
        """
        serie_train= self.serie[start:self.train_len + start]

        # if coef not stationary
        try:
            arima_model= ARIMA(serie_train, order= order)
            arima_fitted= arima_model.fit(disp= False)

        except:
            return {"order": order
                , "metrics": {"aic": np.nan, "bic": np.nan}
                , "confidence interval": np.array([np.nan])
                , "forecast": np.array([np.nan])
                , "start": start
                    }

        aic= arima_fitted.aic
        bic= arima_fitted.bic
        forecasting, std, conf_int= arima_fitted.forecast(self.forecast_range, alpha= self.alpha)

        return {"order": order
            , "metrics": {"aic": aic, "bic": bic}
            , "confidence interval": conf_int
            , "forecast": forecasting
            , "start": start
                }


    def __mean_time_weighted(self, mat_pred_during_time):
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
        del wma, shape_mat_transpose, num_notzero_col, test_vec\
            , vector_pos_nonzero, position_nan, cumsum_notnan\
            , vector_multiplication, n, divisor, i

        return stock_wma


    def __time_looping(self):
        return [self.__select_best_model(i) for i in range(self.num_forecasting)]

    def __order_looping(self, start):
        all_possible_order_permutation= self.__all_3_permutations()
        return [self.__fit(start, ord) for ord in all_possible_order_permutation
                if ord[0] not in self.enforce_complexity["p"]
                and ord[1] not in self.enforce_complexity["d"]
                and ord[2] not in self.enforce_complexity["q"]
                ]

    def __select_best_model(self, start):
        if self.verbose:
            print("Sample {}".format(start))

        all_models= self.__order_looping(start)

        all_criteria= np.array([x["metrics"][self.metric] for x in all_models])
        all_criteria[np.isnan(all_criteria)] = 10e6
        min_index= all_criteria.argmin()

        return all_models[min_index]

    def fit(self):
        return self.__time_looping()


    def forecast(self):
        """
        Warning it calls fit method !

        :return:
        """
        all_forecast= self.fit()

        mat_pred_during_time = np.zeros((self.num_forecasting, self.num_forecasting + self.forecast_range  - 1))

        for i in range(self.num_forecasting):
            pred_t1 = all_forecast[i]["forecast"]  # [1:]

            mat_pred_during_time[i][i:self.forecast_range  + i] = pred_t1  # ajoute la pred dans la matrice


        mean_time_pred = np.nanmean(np.where(mat_pred_during_time != 0, mat_pred_during_time, np.nan), axis=0)
        pred_wm = np.array(self.__mean_time_weighted(mat_pred_during_time))

        return {"mean": mean_time_pred
                , "time_weighted_mean": pred_wm
                }

    def plot(self, forecast_serie, figsize: tuple= (8, 5)) -> None:
        plt.figure(figsize= figsize)

        plt.plot(self.serie, label= "Original serie")
        plt.plot(np.arange(self.train_len, self.len_serie + self.forecast_range, 1)
                 , forecast_serie, label= "Forecast")

        plt.show()
        pass

    def plot_forecast(self):
        pass