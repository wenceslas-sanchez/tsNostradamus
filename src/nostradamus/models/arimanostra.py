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
from scipy.optimize import brute

from typing import Any, Iterable, List, Dict, Tuple

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.api import add_constant
from statsmodels import regression
from statsmodels.graphics.tsaplots import plot_acf

from ..utils.error import (exception_type
, check_is_int
, check_is_in
, check_key_is_in
, check_method_lauched)

from ..models.mother import ArimaMother

class ArimaNostra(ArimaMother):

    def __init__(self, serie, max_order_set, train_len, forecast_range
                 , walk_forward=True, alpha=0.05, metric="aic", verbose=True
                 , enforce_complexity=None):

        super().__init__(serie, max_order_set, train_len, forecast_range
                         , walk_forward=walk_forward, alpha=alpha, metric=metric, verbose=verbose)

        pass


    def __fit(self, start: int, order: Iterable[int]) -> Dict[str, Any]:
        """

        :param start:
        :param order:
        :return:
        """
        serie_train = self.serie[start:self.train_len + start]

        # if coef not stationary
        try:
            arima_model = ARIMA(serie_train, order=order)
            arima_fitted = arima_model.fit(disp=False)

        except:
            return {"order": order
                , "metrics": {"aic": np.nan, "bic": np.nan}
                , "confidence interval": np.array([np.nan])
                , "forecast": np.array([np.nan])
                , "start": start
                    }

        aic = arima_fitted.aic
        bic = arima_fitted.bic
        forecasting, std, conf_int = arima_fitted.forecast(self.forecast_range, alpha=self.alpha)

        return {"order": order
            , "metrics": {"aic": aic, "bic": bic}
            , "confidence interval": conf_int
            , "forecast": forecasting
            , "start": start
                }

    def __fit_optim_arima(self, order: Iterable[int], start: int) -> float:
        """
        """
        serie_train = self.serie[start:self.train_len + start]

        try:
            arima_model = ARIMA(serie_train, order=order)
            arima_fitted = arima_model.fit(disp=False)

            aic = arima_fitted.aic
            bic = arima_fitted.bic
            forecasting, std, conf_int = arima_fitted.forecast(self.forecast_range, alpha=self.alpha)

            # forecast coherence
            max_val = max(self.serie[start:start + self.train_len])
            min_val = min(self.serie[start:start + self.train_len])
            min_bool = sum(forecasting < min_val) == 0
            max_bool = sum(forecasting > max_val) == 0

            if max_bool & min_bool:
                if self.metric == "aic":
                    return aic
                elif self.metric == "bic":
                    return bic
                else:
                    raise ValueError("Select a valid error metric")

            return 10e6

        except:
            return 10e6

    def __generate_grid_from_param(self) -> Tuple[slice]:
        """
        """
        return tuple([slice(self.enforce_complexity[i], self.max_order_set[i], 1) \
                      for i in range(len(self.max_order_set))])

    def __search_for_the_goodone(self, start: int) -> Dict[str, Any]:
        """
        args[0]= ts_churn_volume
        args[1]= i
        args[2]= length_train
        args[3]= forecast_range

        """
        print("Sample {}".format(start))
        grid_params = self.__generate_grid_from_param()

        best_order = brute(self.__fit_optim_arima
                           , ranges=grid_params
                           , args=[start]
                           , finish=None
                           )
        best_order = best_order.astype('int32')

        return self.__fit(start, order=best_order)

    def __time_looping(self) -> List[Dict[str, Any]]:
        return [self.__search_for_the_goodone(i) for i in range(self.num_forecasting)]

    def __error_models(self) -> np.ndarray:
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
        self.fit_ = self.__time_looping()

        self.__error_models()  # compute error
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
        pred_wm = np.array(self.__mean_time_weighted(mat_pred_during_time))

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
        fig.subplots_adjust(hspace= 0.5)

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
        alpha, beta= self.__alpha_beta_coef(x= flatten_all_forecast_error, y= flatten_all_real)

        ax[0, 1].scatter(y = flatten_all_real, x = flatten_all_forecast_error)
        ax[0, 1].plot(flatten_all_forecast_error, alpha + flatten_all_forecast_error * beta, color="red")
        #ax[0, 1].grid()
        ax[0, 1].set_ylabel("Observations")
        ax[0, 1].set_xlabel("Forecast error")
        ax[0, 1].set_title("OLS Obs vs error")

        # autocorr
        plot_acf(flatten_all_forecast_error, lags= self.forecast_range, ax= ax[1, 0])

        # qqplot
        stats.probplot(residuals, plot=plt)

        plt.show()

        pass
