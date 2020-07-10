import pytest
import numpy as np

from nostradamus.preprocessing.tunnel_snake import tunnelSnake

test_data = np.array([1, 2, 4, 6, 4, 5, 3, 7])

def test_fit_transform(test_data= test_data):
    b = np.array([2.5, 2.5, 4.0, 6.0, 4.0, 5.0, 3.0, 6.0])
    ts = tunnelSnake(test_data, 3, 0.5).fit_transform(verbose=False)
    assert np.array_equal(ts, b)

