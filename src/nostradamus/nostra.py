import numpy as np
from statsmodels.tsa.arima_model import ARIMA

def tunnelSnake(serie, shift, threshold):
    """

    :param serie: array
    :param shift: integer, the window wanted for the moving average
    :param threshold: float, define the level
    :return: array, serie without outlier values
    """
    assert type(shift) == int
    assert len(serie) != 0

    def moving_average(serie, shift):
        cumsum_array= np.cumsum(serie)
        cumsum_array[shift:]= cumsum_array[shift:] - cumsum_array[:-shift]
        cumsum_array= cumsum_array[shift-1:]/shift
        return cumsum_array

    def augmented_borne_ma(serie, shift, threshold):
        moving_average_serie= moving_average(serie, shift)
        moving_average_serie= np.append(serie[:-shift-1], moving_average_serie[:-1])

        up_aug_ma= moving_average_serie.copy()
        up_aug_ma[shift:]= up_aug_ma[shift:] * (1 + threshold)

        down_aug_ma = moving_average_serie.copy()
        down_aug_ma[shift:] = down_aug_ma[shift:] * (1 - threshold)

        return up_aug_ma, down_aug_ma;

    # Control serie type and serie content type
    serie= np.array(serie).astype(float)

    print(augmented_borne_ma(serie, shift, threshold))
    up_aug_ma, down_aug_ma= augmented_borne_ma(serie, shift, threshold)

    print(serie)
    boolean_up = serie >= up_aug_ma
    boolean_down = serie < down_aug_ma

    boolean_serie = boolean_up * up_aug_ma + boolean_down * down_aug_ma
    index_to_change = (boolean_serie != 0).nonzero()

    print(boolean_up)
    print(boolean_down)

    treated_serie = serie.copy()
    treated_serie[index_to_change[0]] = boolean_serie[index_to_change[0]]

    return treated_serie



class BasicNostra():

    def __init__(self):
        print("Classe nostra BB")
        pass

    def fit(self, serie, window, params, walk_forward= True):
        pass

    def __all_parameters_arima(self, params):
        """

        Input :
            - perm_list : tuple or list like ;

        Output : list like : each possible permutations from perm_list parameters

        """

        return [[i, j, k]
                for i in range(params[0] + 1)
                for j in range(params[1] + 1)
                for k in range(params[2] + 1)]
