"""
Mother Class for models
"""

class Model():
    """
    Mother class
    """
    def __init__(self, serie, window, forecast_range):
        self.serie= serie
        self.window= window
        self.forecast_range= forecast_range
        pass