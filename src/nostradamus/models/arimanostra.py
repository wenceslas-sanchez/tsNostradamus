import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

import scipy
from scipy.stats import skewnorm
from scipy.optimize import brute

sns.set()
import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima_model import ARIMA

from ..utils.error import (exception_type
, check_is_int
, check_is_in
, check_key_is_in
, check_method_lauched)


class ArimaNostra:

    def __init__(self, serie, max_order_set, train_len, forecast_range
                 , walk_forward=True, alpha=0.05, metric="aic", verbose=True
                 , enforce_complexity=None):
        """

        :param serie:
        :param max_order_set:
        :param train_len:
        :param forecast_range:
        :param walk_forward:
        :param alpha:
        """

        if enforce_complexity is None:
            #enforce_complexity = {"p": [], "d": [], "q": []}
            enforce_complexity= [0, 0, 0]

        self.serie = serie
        self.max_order_set = max_order_set
        self.len_serie = len(serie)
        self.train_len = train_len
        self.forecast_range = forecast_range
        self.walk_forward = walk_forward
        self.alpha = alpha
        self.metric = metric
        self.num_forecasting = self.len_serie - self.train_len + 1
        self.verbose = verbose
        self.enforce_complexity = enforce_complexity

        self.fit_= None
        self.forecast_= None
        self.method= None
        self.error_by_models= None
        self.error_by_periods= None
        self.metric_error= None

        # Control type
        exception_type(self.serie, (list, tuple, np.ndarray))
        check_is_int(self.train_len)
        check_is_int(self.forecast_range)

        check_is_in(self.metric, ["aic", "bic"])
        #check_key_is_in(["p", "d", "q"], self.enforce_complexity)
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

    def __fit_optim_arima(self, order, start):
        """
        """
        serie_train = self.serie[start:self.train_len + start]

        try:
            arima_model = ARIMA(serie_train, order=order)
            arima_fitted = arima_model.fit(disp=False)

            aic = arima_fitted.aic
            bic = arima_fitted.bic
            forecasting, std, conf_int = arima_fitted.forecast(self.forecast_range, alpha=self.alpha)

            max_val = max(self.serie[start:start + self.train_len])
            min_val = min(self.serie[start:start + self.train_len])
            min_bool = sum(forecasting < min_val) == 0  # ne prend que les modeles qui predisent des churn positifs
            max_bool = sum(forecasting > max_val) == 0

            if max_bool & min_bool:
                return aic

            return 10e6

        except:
            return 10e6

    def __generate_grid_from_param(self):
        """
        """
        return tuple([slice(self.enforce_complexity[i], self.max_order_set[i], 1) \
                      for i in range(len(self.max_order_set))])

    def __search_for_the_goodone(self, start):
        """
        args[0]= ts_churn_volume
        args[1]= i
        args[2]= length_train
        args[3]= forecast_range

        """
        print("Sample {}".format(start))
        grid_params= self.__generate_grid_from_param()

        best_order= brute(self.__fit_optim_arima
                           , ranges= grid_params
                           , args= [start]
                           , finish= None
                           )

        best_order= best_order.astype('int32')

        return self.__fit(start, order=best_order)



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
        del wma, shape_mat_transpose, num_notzero_col, test_vec \
            , vector_pos_nonzero, position_nan, cumsum_notnan \
            , vector_multiplication, n, divisor, i

        return stock_wma

    def __time_looping(self):
        return [self.__search_for_the_goodone(i) for i in range(self.num_forecasting)]


    def __error_models(self):
        # error for each model

        error_list = []
        for i in range(self.num_forecasting - self.forecast_range):
            pred_t1 = self.fit_[i]["forecast"]
            real_values = self.serie[self.train_len + i: self.train_len + self.forecast_range + i]
            error_list.append(pred_t1 - real_values)

        self.error_by_models = np.array(error_list)
        return self.error_by_models

    def fit(self):
        self.fit_= self.__time_looping()

        self.__error_models() # compute error
        return self.fit_

    def forecast(self, method= "tw_mean"):
        """
        Forecast_serie = return de fit !

        :return:
        """
        self.method= method

        mat_pred_during_time = np.zeros((self.num_forecasting, self.num_forecasting + self.forecast_range - 1))

        for i in range(self.num_forecasting):
            pred_t1 = self.fit_[i]["forecast"]  # [1:]

            mat_pred_during_time[i][i:self.forecast_range + i] = pred_t1  # ajoute la pred dans la matrice

        mean_time_pred = np.nanmean(np.where(mat_pred_during_time != 0, mat_pred_during_time, np.nan), axis=0)
        pred_wm = np.array(self.__mean_time_weighted(mat_pred_during_time))

        self.forecast_= {"mean": mean_time_pred
                            , "tw_mean": pred_wm
                        }

        return self.forecast_


    def error(self, error_type= "mae"):
        # forecast error
        if error_type == "mae":
            self.error_by_periods= np.mean(np.abs(self.error_by_models), axis= 0)
            self.metric_error= np.mean(self.error_by_periods)
            return self.metric_error

        elif error_type == "rmse":
            self.error_by_periods= np.sqrt(np.mean(np.square(self.error_by_models), axis= 0))
            self.metric_error= np.mean(self.error_by_periods)
            return self.metric_error
        # add other error computing methods
        else:
            raise ValueError("Select ")
        pass

    def plot_models(self, figsize: tuple = (8, 5), random_state= 55, legend= True) -> None:

        plt.figure(figsize=figsize)

        plt.plot(self.serie, label="Original serie")

        # Generate unique color for each model we build
        np.random.seed(random_state)
        for i in range(self.num_forecasting):
            r = lambda: np.random.randint(0, 255)
            hex_number = '#%02X%02X%02X' % (r(), r(), r())

            pred_t1 = self.fit_[i]
            plt.plot(np.arange(self.train_len + i, self.train_len + self.forecast_range + i, 1)
                     , pred_t1["forecast"], c= hex_number
                     , label= "Sample {} : Order {}".format(i, pred_t1["order"]))

        if legend:
            plt.legend()
        plt.show()
        pass

    def plot(self, figsize: tuple = (8, 5)) -> None:
        # Need to forecast beafore
        check_method_lauched(self.method, None)

        plt.figure(figsize=figsize)

        serie_forecast= self.forecast_[self.method]

        plt.plot(self.serie, label="Original serie")
        plt.plot(np.arange(self.train_len, self.len_serie + self.forecast_range, 1)
                 , serie_forecast, label="Forecast")

        plt.show()
        pass


    def plot_error_by_period(self, figsize: tuple = (8, 5), error_dist= None, bins= 10) -> None:

        error_type= "mae"
        if self.error_by_periods is None:
            self.error(error_type)

        if error_dist is None:
            error_dist= "no"

        ax = plt.figure(figsize= figsize).gca()

        # hist
        if error_dist == "hist":
            error_matrix_period= np.abs(self.error_by_models).T

            for x in range(self.forecast_range):
                weights= np.ones_like(error_matrix_period[x]) / len(error_matrix_period[x])
                ax.hist(error_matrix_period[x], bins= bins, bottom= x, orientation= "horizontal"
                        , weights= weights, color= "purple", alpha= 0.3)

            plt.xlim((-0.1, self.forecast_range))

        elif error_dist == "boxplot":
            error_matrix_period = np.abs(self.error_by_models).T
            error_matrix_period_list= []
            for x in range(self.forecast_range):
                error_matrix_period_list.append(error_matrix_period[x])

            ax.boxplot(error_matrix_period_list, positions= np.arange(0, self.forecast_range, 1).astype(int))

        elif error_dist == "no":
            pass

        else:
            raise ValueError()

        ax.plot(np.arange(0, self.forecast_range, 1).astype(int), self.error_by_periods, color= "purple")

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Period")
        plt.ylabel(error_type)
        plt.show()

        pass

    def plot_forecast(self, figsize: tuple = (8, 5)) -> None:
        # only ploting unobservable values
        plt.figure(figsize=figsize)

        serie_forecast = self.forecast_[self.method]

        plt.plot(self.serie, label="Original serie")
        plt.plot(np.arange(self.len_serie, self.len_serie + self.forecast_range, 1)
                 , serie_forecast[-self.forecast_range:], label="Forecast")

        plt.show()

        pass

    def plot_diagnostic(self, bins= 25):
        # error
        residuals= self.error_by_models.flatten()
        norm_residuals= (residuals - np.mean(residuals))/np.std(residuals)

        fig, ax= plt.subplots(nrows= 2, ncols= 1, figsize= (15, 10))

        # Histogram
        mu, std = scipy.stats.norm.fit(residuals)

        ax[0].hist(residuals, bins=bins, density=True, alpha=0.6, color='purple', label="Données")

        # Plot le PDF.
        xmin, xmax = ax[0].get_xlim()
        X = np.linspace(xmin, xmax)

        ax[0].plot(X, scipy.stats.norm.pdf(X, mu, std), label="Normal Distribution")
        ax[0].plot(X, skewnorm.pdf(X, *skewnorm.fit(residuals)), color='black', label="Skewed Normal Distribution")

        mu, std = scipy.stats.norm.fit(residuals)
        sk = scipy.stats.skew(residuals)

        title2 = "Moments mu: {}, sig: {}, sk: {}".format(round(mu, 4), round(std, 4), round(sk, 4))
        ax[0].set_ylabel("Fréquence", rotation=90)
        ax[0].set_title(title2)
        ax[0].legend()

        # qqplot

        plt.show()
        pass
