"""
Dependencies
"""

import numpy as np
import pandas as pd
import math
#from random import random
import datetime
#from dateutil.relativedelta import relativedelta

# models
from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.arima_model import ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
#from statsmodels.tsa.statespace.sarimax import SARIMAXResults


"""
Methods
"""


def adjust_datetime(df, col_date, format_adj="%Y/%m/%d"):
    """
    Transform a datetime like 2020-05-31 23:00:00 to 2020-06-01 (end of M-1 to start of M)

    Input :
        - df : Dataframe like
        - col_date : string like ; name of the column containing the datetime object to adjust
        - format_adj : string like ; date format displayed

    Output : Transformed Dataframe like

    Handling error : column name "DAT_PHTO" is required, with datetime64[ns] like format
    """

    df["DAT_PHTO"] = df["DAT_PHTO"] + datetime.timedelta(days=1)
    df["DAT_PHTO"] = pd.to_datetime(df["DAT_PHTO"].dt.strftime("%Y/%m/%d"), format="%Y/%m/%d")
    df = df.sort_values(["DAT_PHTO"], ascending=True).reset_index(drop=True)

    return df


def arima_model_error_computing(ts, start, hist_train_size, valid, order):
    """
    From parameters set, train an ARIMA model, and produce a kind of out sample forecasting.
    Indeed, we are training our model on an train dataset, and using another one to compare forecastings.
    Is computed an ts_valid, which is a part of our time serie data, which is not in the train dataset
        , and is the folowing serie of the train serie.
    It allows us to compute forecasing error.

    Input :
        - ts : array like ; the time series we want to modelize (need to be ordered)
        - start : int ; first index of our time serie
        - hist_train_size : int ; last index of our time serie. It describe also the length of the train data.
        - valid : int ; how many point we predict with our trained model
        - order : list like ; contains the three parameters (p, d, q) requisite for the ARIMA(P, D, Q)

    Output : tuple like containing :
        - if output 1 : full of NAN, len(output) == 4
        - if output 2 : order = tuple like ; ARIMA parameters, and full of NAN, len(output) == 3 !!
        - if output 3 :
                        - order : tuple likeA ; RIMA parameters
                        - error_valid : array like ; containing forecasting error (from out sample forecasting)
                        - aic : float like
                        - conf_int : array like
                        - pred : array like ; containing prediction from out sample forecasting
                        Warning, len(output) == 5

    Handling errors : no error should be generated, but check output
    """

    ts_train = ts[start:hist_train_size + start]
    ts_valid = ts[[i for i in np.arange(hist_train_size + start, hist_train_size + start + valid)]]

    # if coef not stationary
    try:
        arima_model = ARIMA(ts_train, order=order)
        arima_fitted = arima_model.fit(disp=False)
    except:
        try:  # if MA coef not invertible
            if order[1] < 1:
                order[1] = order[1] + 1  # parameter D
                arima_model = ARIMA(ts_train, order=order)
                arima_fitted = arima_model.fit(disp=False)
            else:
                return (np.nan, np.array([np.nan]), np.nan)  # output 1
        except:
            return (np.nan, np.array([np.nan]), np.nan)  # output 1

    #aic = ARMAResults.aic(arima_fitted)
    aic= arima_fitted.aic
    pred, std, conf_int = arima_fitted.forecast(valid, alpha=0.05)

    error_valid = ts_valid - pred

    if math.isnan(error_valid[0]):
        return (order, np.array([np.nan]), np.nan)  # output 2
    else:
        return (order, error_valid, aic, conf_int, pred)  # output 3
    pass


def all_3_permutations(perm_list):
    """

    Input :
        - perm_list : tuple or list like ;

    Output : list like : each possible permutations from perm_list parameters

    """

    return [[i, j, k]
            for i in range(perm_list[0] + 1)
            for j in range(perm_list[1] + 1)
            for k in range(perm_list[2] + 1)]


def outlier_treat_MA(df, col_date, col_val, window, threshold):
    """
    Allow to smooth a time series, with a moving average. It allows us to eliminate some outliers (extreme values)

    Input :
        - df : dataframe like, containing at least 2 columns
        - col_date : string like, name of the datetime column
        - col_val : string like, name of the serie column
        - window : int like, window size for moving average

    Output : array like : smoothed serie

    The goal is to predict predictable event and sometimes incidents derange our goal. Because those events are not predictable.
    With a moving average smoothing, we limit the amplitude our signal can go to delete unpredictable incidents.
    """
    df = df.sort_values([col_date])
    df = df.reset_index(drop=True)

    data_rolled_vol = df[col_val].rolling(window)
    rol_mean_vol_top = np.array([0] + np.array(data_rolled_vol.mean() * (1 + threshold)).tolist())[:-1]

    rol_mean_vol_bot = np.array([0] + np.array(data_rolled_vol.mean() * (1 - threshold)).tolist())[:-1]

    rol_mean_vol_top[0:window] = df[col_val].values[0:window]
    rol_mean_vol_bot[0:window] = df[col_val].values[0:window]

    churn_vol = np.array([x for x in df[col_val].values])
    churn_vol

    booled_top = churn_vol >= rol_mean_vol_top
    booled_bot = churn_vol < rol_mean_vol_bot

    tested = booled_top * rol_mean_vol_top + booled_bot * rol_mean_vol_bot
    tested

    index_to_change = (tested != 0).nonzero()

    tested[index_to_change[0]]

    churn_trans = churn_vol.copy()

    churn_trans[index_to_change[0]] = tested[index_to_change[0]]

    return churn_trans


def wm_prediction(mat_pred_during_time):
    """


    """
    shape_mat_transpose = mat_pred_during_time.T.shape[0]

    stock_wma = []

    num_notzero_col = np.count_nonzero(mat_pred_during_time, axis=0)

    #
    for i in range(shape_mat_transpose):
        test_vec = mat_pred_during_time.T[i]

        vector_pos_nonzero = ~(test_vec == 0) * 1  # recupere

        vector_pos_nonzero = np.where(vector_pos_nonzero != 0, vector_pos_nonzero, np.nan)  # remplace les 0 pas des nan

        position_nan = ~np.isnan(vector_pos_nonzero)  # je prend la position des valeurs non nan
        cumsum_notnan = np.cumsum(vector_pos_nonzero[position_nan])  # cumsum les valeurs non nan

        vector_pos_nonzero[position_nan] = cumsum_notnan  # integre la cumsum dans le vecteur

        vector_multiplication = np.nan_to_num(vector_pos_nonzero, 0)  # remplace les nan par des 0

        n = num_notzero_col

        divisor = n[i] * (n[i] + 1) / 2

        # weighted average
        wma = np.sum((test_vec * vector_multiplication) / divisor)

        stock_wma.append(wma)

    return stock_wma


def arima_model_forecast_computing(ts, start, hist_train_size, forecast_range, order):
    """

    Input :
        - ts : array like ; the time series we want to modelize
        - start : int ; first index of our time serie
        - hist_train_size : int ; last index of our time serie. It describe also the length of the train data.
        - valid : int ; how many point we predict with our trained model
        - order : list like ; contains the three parameters (p, d, q) requisite for the ARIMA(P, D, Q)

    Output : list like containing :
        -
        -
        -
        -

    Handling errors :
    """

    ts_train = ts[start:hist_train_size + start]

    # if coef not stationary
    try:
        arima_model = ARIMA(ts_train, order=order)
        arima_fitted = arima_model.fit(disp=False)
    except:
        try:  # if MA coef not invertible
            if order[1] < 1:
                order[1] = order[1] + 1  # parameter D
                arima_model = ARIMA(ts_train, order=order)
                arima_fitted = arima_model.fit(disp=False)
            else:
                return (np.nan, np.nan, np.array([np.nan]), np.array([np.nan]))
        except:
            return (np.nan, np.nan, np.array([np.nan]), np.array([np.nan]))

    #aic = ARMAResults.aic(arima_fitted)
    aic = arima_fitted.aic
    pred, std, conf_int = arima_fitted.forecast(forecast_range, alpha=0.05)

    return (order, aic, conf_int, pred)


def sarima_model_error_computing(ts, start, hist_train_size, valid, order):
    """

    """
    arma_order = order[0]
    s_order = order[1]

    ts_train = ts[start:hist_train_size + start]
    ts_valid = ts[[i for i in np.arange(hist_train_size + start, hist_train_size + start + valid)]]

    try:
        sarima_model = SARIMAX(ts_train, order=arma_order, seasonal_order=s_order
                               , enforce_stationarity=False, enforce_invertibility=False)
        sarimax_fitted = sarima_model.fit(disp=True)

        #bic = SARIMAXResults.bic(sarimax_fitted)
        bic= sarimax_fitted.bic

        pred = sarimax_fitted.forecast(valid, alpha=0.05)
        error = ts_valid - pred
        return (order, error, bic, pred)
    except:
        return (np.nan, [np.nan], np.nan, np.nan)
    pass


def sarima_model_forecast_computing(ts, start, hist_train_size, valid, order):
    """

    """
    arma_order = order[0]
    s_order = order[1]

    ts_train = ts[start:hist_train_size + start]

    try:
        sarima_model = SARIMAX(ts_train, order=arma_order, seasonal_order=s_order
                               , enforce_stationarity=False, enforce_invertibility=False)
        sarimax_fitted = sarima_model.fit(disp=True)

        bic = SARIMAXResults.bic(sarimax_fitted)
        bic = sarimax_fitted.bic

        pred = sarimax_fitted.forecast(valid, alpha=0.05)
        return (order, bic, pred)
    except:
        return (np.nan, np.nan, np.array([np.nan]), np.array([np.nan]))
    pass


def all_sarima_permutations(perm_list):
    """
    perm_list : tuple or list like ;
    """
    return [([i, j, k], [n, m, x, perm_list[6]])
            for i in range(perm_list[0] + 1)
            for j in range(perm_list[1] + 1)
            for k in range(perm_list[2] + 1)
            for n in range(perm_list[3] + 1)
            for m in range(perm_list[4] + 1)
            for x in range(perm_list[5] + 1)
            #             for y in range(perm_list[6]+1)
            ]