import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set()
import warnings

warnings.filterwarnings("ignore")

import scipy
from scipy.stats import skewnorm
from scipy import stats

from typing import Any, Iterable, List, Dict, Tuple

from statsmodels.api import add_constant
from statsmodels import regression
from statsmodels.graphics.tsaplots import plot_acf

from tqdm import tqdm

from ..utils.error import (exception_type
, check_is_int
, check_is_in
, check_key_is_in
, check_method_lauched)

"""
Mother Class for models
"""


class MotherModel():
    """
    Mother class
    """

    def __init__(self):
        pass



class ArimaMother(MotherModel):
    """
    Mother class of ARIMA like models
    """

    def __init__(self, serie: np.ndarray, max_order_set: Iterable[int], train_len: int, forecast_range: int
                 , walk_forward: bool = True, alpha: float = 0.05, metric: str = "aic", verbose: bool = True
                 , enforce_complexity: Iterable[int] = None):
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

        if self.enforce_complexity is None:
            # enforce_complexity = {"p": [], "d": [], "q": []}
            self.enforce_complexity = [0 for i in range(len(self.max_order_set))]


        self.fit_ = None
        self.forecast_ = None
        self.method = None
        self.error_by_models = None
        self.error_by_periods = None
        self.metric_error = None
        self.real_and_forcecast = None
        self.fit_ = None

        # import method from Mother class (plotting functions at least)
        super().__init__()

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

    def search_for_the_goodone(self, start) -> Dict[str, Any]:
        return {"1": 1}

    def time_looping(self) -> List[Dict[str, Any]]:
        good_list= []
        # return [self.search_for_the_goodone(i) for i in range(self.num_forecasting)]
        for i in tqdm(range(self.num_forecasting), desc= "Computing"):
            good_list.append(self.search_for_the_goodone(i))

        return good_list

    def error_models(self) -> np.ndarray:
        # error for each model

        error_list = []
        real_and_forecast = []
        for i in range(self.num_forecasting - self.forecast_range):
            pred_t1 = self.fit_[i]["forecast"]
            real_values = self.serie[self.train_len + i: self.train_len + self.forecast_range + i]
            error_list.append(pred_t1 - real_values)
            real_and_forecast.append((real_values, pred_t1))

        self.error_by_models = np.array(error_list)
        self.real_and_forcecast = np.array(real_and_forecast)

        return self.error_by_models

    def fit(self) -> List[Dict[str, Any]]:
        self.fit_ = self.time_looping()

        self.error_models()  # compute error

        return self.fit_

    def forecast(self, method: str = "tw_mean") -> Dict[str, np.ndarray]:
        """
        Forecast_serie = return de fit !

        :return:
        """
        self.method = method

        mat_pred_during_time = np.zeros((self.num_forecasting, self.num_forecasting + self.forecast_range - 1))

        for i in range(self.num_forecasting):
            pred_t1 = self.fit_[i]["forecast"]  # [1:]

            mat_pred_during_time[i][i:self.forecast_range + i] = pred_t1  # ajoute la pred dans la matrice

        mean_time_pred = np.nanmean(np.where(mat_pred_during_time != 0, mat_pred_during_time, np.nan), axis=0)
        pred_wm = np.array(self.mean_time_weighted(mat_pred_during_time))

        self.forecast_ = {"mean": mean_time_pred
            , "tw_mean": pred_wm
                          }

        return self.forecast_

    def error(self, error_type: str = "mae") -> np.ndarray:
        # forecast error
        if error_type == "mae":
            self.error_by_periods = np.mean(np.abs(self.error_by_models), axis=0)
            self.metric_error = np.mean(self.error_by_periods)
            return self.metric_error

        elif error_type == "rmse":
            self.error_by_periods = np.sqrt(np.mean(np.square(self.error_by_models), axis=0))
            self.metric_error = np.mean(self.error_by_periods)
            return self.metric_error
        # add other error computing methods
        else:
            raise ValueError("Select a valid metric error")
        pass

    def plot_models(self, figsize: Tuple = (8, 5), random_state: int = 55, legend: bool = True) -> None:

        plt.figure(figsize=figsize)

        plt.plot(self.serie, label="Original serie")

        # Generate unique color for each model we build
        np.random.seed(random_state)
        for i in range(self.num_forecasting):
            r = lambda: np.random.randint(0, 255)
            hex_number = '#%02X%02X%02X' % (r(), r(), r())

            pred_t1 = self.fit_[i]
            plt.plot(np.arange(self.train_len + i, self.train_len + self.forecast_range + i, 1)
                     , pred_t1["forecast"], c=hex_number
                     , label="Sample {} : Order {}".format(i, pred_t1["order"]))

        if legend:
            plt.legend()

        plt.show()
        pass

    def plot(self, figsize: Tuple = (8, 5)) -> None:
        # Need to forecast beafore
        check_method_lauched(self.method, None)

        plt.figure(figsize=figsize)

        serie_forecast = self.forecast_[self.method]

        plt.plot(self.serie, label="Original serie")
        plt.plot(np.arange(self.train_len, self.len_serie + self.forecast_range, 1)
                 , serie_forecast, label="Forecast")

        plt.show()
        pass

    def __type_error_matrix_period(self, error_type: str = "mae") -> np.ndarray:
        if error_type == "mae":
            return np.abs(self.error_by_models).T
        else:
            raise ValueError("Only MAE accepted")

    def plot_error_by_period(self, figsize: Tuple = (8, 5), error_dist: str = None, bins: int = 10
                             , error_type: str = "mae") -> None:

        self.error(error_type)  # not mae nor rmse because can't use mean ofr hist or boxplot

        if error_dist is None:
            error_dist = "no"

        ax = plt.figure(figsize=figsize).gca()

        # hist
        error_matrix_period = self.__type_error_matrix_period(error_type)

        if error_dist == "hist":

            for x in range(self.forecast_range):
                weights = np.ones_like(error_matrix_period[x]) / len(error_matrix_period[x])
                ax.hist(error_matrix_period[x], bins=bins, bottom=x, orientation="horizontal"
                        , weights=weights, color="purple", alpha=0.3)

            plt.xlim((-0.1, self.forecast_range))

        elif error_dist == "boxplot":

            error_matrix_period_list = []
            for x in range(self.forecast_range):
                error_matrix_period_list.append(error_matrix_period[x])

            ax.boxplot(error_matrix_period_list, positions=np.arange(0, self.forecast_range, 1).astype(int))


        elif error_dist == "no":
            pass

        else:
            raise ValueError()

        ax.plot(np.arange(0, self.forecast_range, 1).astype(int), self.error_by_periods, color="purple")

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Period")
        plt.ylabel(error_type)
        plt.show()

        pass

    def plot_forecast(self, figsize: Tuple = (8, 5)) -> None:
        # only ploting unobservable values
        plt.figure(figsize=figsize)

        serie_forecast = self.forecast_[self.method]

        plt.plot(self.serie, label="Original serie")
        plt.plot(np.arange(self.len_serie, self.len_serie + self.forecast_range, 1)
                 , serie_forecast[-self.forecast_range:], label="Forecast")

        plt.show()

        pass

    def __alpha_beta_coef(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x = add_constant(x)
        model = regression.linear_model.OLS(y, x).fit()

        alpha, beta = model.params[0], model.params[1]
        return alpha, beta

    def plot_diagnostic(self, bins: int = 25) -> None:
        # error
        residuals = self.error_by_models.flatten()
        norm_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
        flatten_all_real = np.array([self.real_and_forcecast[i][0] for i in range(self.num_forecasting \
                                                                                  - self.forecast_range)]).flatten()

        flatten_all_forecast_error = np.array([self.real_and_forcecast[i][1] - self.real_and_forcecast[i][0]
                                               for i in range(self.num_forecasting - self.forecast_range)]).flatten()

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5)

        # Histogram
        mu, std = scipy.stats.norm.fit(residuals)

        ax[0, 0].hist(residuals, bins=bins, density=True, alpha=0.6, color='purple', label="Données")

        # Plot le PDF.
        xmin, xmax = ax[0, 0].get_xlim()
        X = np.linspace(xmin, xmax)

        ax[0, 0].plot(X, scipy.stats.norm.pdf(X, mu, std), label="Normal Distribution")
        ax[0, 0].plot(X, skewnorm.pdf(X, *skewnorm.fit(residuals)), color='black', label="Skewed Normal Distribution")

        mu, std = scipy.stats.norm.fit(residuals)
        sk = scipy.stats.skew(residuals)

        title2 = "Moments mu: {}, sig: {}, sk: {}".format(round(mu, 4), round(std, 4), round(sk, 4))
        ax[0, 0].set_ylabel("Fréquence", rotation=90)
        ax[0, 0].set_title(title2)
        ax[0, 0].legend()

        # OLS
        alpha, beta = self.__alpha_beta_coef(x=flatten_all_forecast_error, y=flatten_all_real)

        ax[0, 1].scatter(y=flatten_all_real, x=flatten_all_forecast_error)
        ax[0, 1].plot(flatten_all_forecast_error, alpha + flatten_all_forecast_error * beta, color="red")
        # ax[0, 1].grid()
        ax[0, 1].set_ylabel("Observations")
        ax[0, 1].set_xlabel("Forecast error")
        ax[0, 1].set_title("OLS Obs vs error")

        # autocorr
        plot_acf(flatten_all_forecast_error, lags=self.forecast_range, ax=ax[1, 0])

        # qqplot
        stats.probplot(residuals, plot=plt)

        plt.show()

        pass
