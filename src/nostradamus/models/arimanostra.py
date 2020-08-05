import numpy as np
import warnings

warnings.filterwarnings("ignore")
from scipy.optimize import brute
from typing import Any, Iterable, List, Dict, Tuple
from statsmodels.tsa.arima_model import ARIMA
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

    def search_for_the_goodone(self, start: int) -> Dict[str, Any]:
        """
        args[0]= ts_churn_volume
        args[1]= i
        args[2]= length_train
        args[3]= forecast_range

        """
        print("Sample {}".format(start))
        grid_params = self.generate_grid_from_param()

        best_order = brute(self.__fit_optim_arima
                           , ranges=grid_params
                           , args=[start]
                           , finish=None
                           )
        best_order = best_order.astype('int32')

        return self.__fit(start, order=best_order)
