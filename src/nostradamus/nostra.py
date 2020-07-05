import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA


def exception_type(arg, typed):
    if isinstance(arg, typed):
        pass
    else:
        raise Exception("Wrong Type")


class tunnelSnake():

    def __init__(self, serie, shift, threshold):
        self.serie= serie
        self.shift= shift
        self.threshold= threshold

        # Control type
        exception_type(self.serie, (list, tuple, np.ndarray))
        exception_type(self.threshold, (int, float))
        exception_type(self.shift, int)
        #exception_type(verbose, bool)

        assert len(self.serie) > self.shift
        pass

    def __moving_average(self):
        cumsum_array= np.cumsum(self.serie)
        cumsum_array[self.shift:]= cumsum_array[self.shift:] - cumsum_array[:-self.shift]
        cumsum_array= cumsum_array[self.shift-1:]/self.shift
        return cumsum_array

    def __augmented_borne_ma(self):
        moving_average_serie= self.__moving_average()
        moving_average_serie= np.append(self.serie[0:self.shift], moving_average_serie[:-1])

        up_aug_ma= moving_average_serie.copy()
        up_aug_ma[self.shift:]= up_aug_ma[self.shift:] * (1 + self.threshold)

        down_aug_ma = moving_average_serie.copy()
        down_aug_ma[self.shift:] = down_aug_ma[self.shift:] * (1 - self.threshold)

        return up_aug_ma, down_aug_ma;

    def fit_transform(self, verbose= True):
        """

        :param verbose:
        :return: array, serie without outlier values
        """
        # Control serie type and serie content type
        self.serie = np.array(self.serie).astype(float)

        up_aug_ma, down_aug_ma = self.__augmented_borne_ma()

        # Which values are sup / inf to the augmented MA
        # and get the augmented MA value when it's True
        boolean_up = self.serie >= up_aug_ma
        boolean_down = self.serie < down_aug_ma
        boolean_serie = boolean_up * up_aug_ma + boolean_down * down_aug_ma

        # Get the index of values we have to change
        index_to_change = (boolean_serie != 0).nonzero()

        if verbose:
            # We removed the shift first values, because they are always into index_to_change
            print("Values at place {} were changed.".format(index_to_change[0][self.shift:]))

        # Replace from inital array
        treated_serie = self.serie.copy()
        treated_serie[index_to_change[0]] = boolean_serie[index_to_change[0]]

        return treated_serie

    def plot(self, figsize= (8, 5)):
        sns.set()
        up_aug_ma, down_aug_ma = self.__augmented_borne_ma()

        plt.figure(figsize= figsize)

        plt.plot(self.fit_transform(verbose= False), label= "Transformed serie"
                 , c= "darkblue", linestyle= "--", linewidth= 0.9)
        plt.plot(self.serie, label= "Original serie", c= "black")
        plt.plot(up_aug_ma, label= "Top boundary", c= "orange")
        plt.plot(down_aug_ma, label= "Down boundary", c= "orange")

        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.legend()

        plt.show()
        pass



class ArimaNostra():

    def __init__(self):
        pass

    def __all_parameters_arima(self, params):
        """

        :param params:
        :return:
        """
        return [[i, j, k]
                for i in range(params[0] + 1)
                for j in range(params[1] + 1)
                for k in range(params[2] + 1)]

    def __arima_model_forecast_computing(self, serie, start, window, forecast_range, order, alpha= 0.05):
        """

        :param start:
        :param hist_train_size:
        :param forecast_range:
        :param order:
        :return:
        """
        ts_train= serie[start:window + start]

        # In case of failed
        nan_return= np.nan, np.nan, np.array([np.nan]), np.array([np.nan]);

        # If coef not stationary
        try:
            arima_model = ARIMA(ts_train, order=order)
            arima_fitted = arima_model.fit(disp=False)
        except:
            try:  # If MA coef not invertible
                if order[1] < 1:
                    order[1] = order[1] + 1  # parameter D
                    arima_model = ARIMA(ts_train, order=order)
                    arima_fitted = arima_model.fit(disp=False)
                else:
                    return nan_return
            except:
                return nan_return

        aic = arima_fitted.aic
        forecast, std, conf_int = arima_fitted.forecast(forecast_range, alpha= alpha)

        return order, aic, conf_int, forecast;

    def fit(self):
        """

        :return:
        """
        pass



class _procedure_train():
    """
    Mother class
    """
    def __init__(self, serie, window, forecast_range):
        self.serie= serie
        self.window= window
        self.forecast_range= forecast_range
        pass

    def walk_forward_train(self):
        pass

    def stacking_train(self):
        pass
