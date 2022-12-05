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
        """~5-6x times faster than pure pandas `run` method"""
        cumsum_vec = np.cumsum(np.insert(data, 0, 0, axis=0), axis=0)
        np_sma = (
            cumsum_vec[self._window :] - cumsum_vec[: -self._window]
        ) / self._window

        if data.ndim == 1:
            nans_shape = self._window - 1
        else:
            nans_shape = (self._window - 1, data.shape[1])

        self._nans = np.full(nans_shape, np.NaN)

        return np.concatenate(
            [self._nans, np_sma]  # adding NaN values to get pandas functionality
        )


class ExponentialMovingAverage(Indicator):
    name = "ema_indicator"
    description = """
    Exponential moving average indicator. Formula:
    y_{0} = x_{0}; 
    y_{i} = alpha * x_{i} + (1-alpha) * y_{i-1};
    where 0 < alpha < 1 and specified directly.
    
    So... If you want special handling - handle it yourself.
    Exaxmple:
    Let window = 10 ==>
    if window_type is `span`     | (>= 0)         ===>   alpha = 2/(window + 1)
    if window_type is `com`      | (>= 1)         ===>   alpha = 1/(window + 1)
    if window_type is `halflife` | (> 0)          ===>   alpha = 1 - e^{-ln(2) / window}
    if window_type is `alpha`    | (>= 1)         ===>   alpha = 1/window
    
    compute alpha with given method
    """

    def __init__(self, alpha: float):
        assert 0 < alpha <= 1
        self._alpha = alpha
        self._k = 1 - self._alpha

    @staticmethod
    def compute_alpha(window: int, window_type: str = "span"):
        assert window >= 2
        assert isinstance(window, int)

        if window_type == "span":
            return 2 / (window + 1)
        elif window_type == "com":
            return 1 / (window + 1)
        elif window_type == "halftime":
            return 1 - np.exp(-np.log(2) / window)
        elif window_type == "alpha":
            return 1 / window

        raise NotImplementedError(
            f"Incorrect window type or not implemented {window_type=}"
        )

    def run(self, data: pd.Series | pd.DataFrame):
        return data.ewm(alpha=self._alpha, adjust=False).mean()

    def np_run(self, data: np.ndarray):
        if data.ndim == 1:
            return self.run(pd.Series(data)).to_numpy()  # Placeholder, cuz it's faster
        return self.run(pd.DataFrame(data)).to_numpy()  # Placeholder, cuz it's faster

        # TODO Think about it. It's slower ~3.5x times with N=100 and ~1000x slower with N=10**6 than pandas version. In pandas cpython implementaition they use functools.partial, maybe this should be better. Don't wnat to use numba.
        # ema = np.empty(data.shape)
        # ema[0] = data[0]
        # for i in range(1, data.shape[0]):
        #     ema[i] = self._alpha * data[i] + self._k * ema[i-1]

        # return ema
