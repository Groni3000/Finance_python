import pandas as pd, numpy as np
from technical_indicators.interfaces import Indicator


class SimpleMovingAverage(Indicator):
    name = "sma_indicator"
    description = "Simple moving average indicator."

    def __init__(self, window: int = 2):
        self._window = window

    def run(self, data: pd.Series | pd.DataFrame):
        return data.rolling(self._window).mean()

    def np_run(self, data: np.ndarray):
        """~6x times faster than pure pandas `run` method"""
        cumsum_vec = np.cumsum(np.insert(data, 0, 0, axis=0), axis=0)
        np_sma = (
            cumsum_vec[self._window :] - cumsum_vec[: -self._window]
        ) / self._window

        if len(data.shape) == 1:
            self._nans = np.full(self._window - 1, np.NaN)
        else:
            self._nans = np.full((self._window - 1, data.shape[1]), np.NaN)

        return np.concatenate(
            [self._nans, np_sma]
        )  # adding NaN values to get pandas functionality
