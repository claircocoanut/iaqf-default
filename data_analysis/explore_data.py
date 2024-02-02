import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from matplotlib import colors
import matplotlib.pyplot as plt


def hurst_exponent(x, max_lag=20):
    """Returns the Hurst Exponent of the time series"""

    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [x.diff(lag).std() for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def calc_rolling_stat(df: pd.DataFrame, month: int = 3) -> pd.DataFrame:
    """
    Calculate rolling period aggregate statistics

    Parameters
    ----------
    df: pd.DataFrame for price
        Columns: tickers
        Index: DatatimeIndex
    month: int
        Number of month as rolling window

    Returns
    -------
    pd.DataFrame
        Columns: [ret, std, ret_max, ret_min] * tickers
            ret: end-to-end return
            std: close-to-close volatility
            ret_max: maximum earn during the period
            ret_min: maximum loss during the period
        Index: DatetimeIndex
    """

    def ret(x): return np.log(x[-1] / x[0])

    def vol(x): return x[:-1].std()

    def ret_max(x): return np.log(x.max() / x[0])

    def ret_min(x): return np.log(x.min() / x[0])

    n = month * 20 + 1
    ret = df.rolling(f"{n}D").aggregate([ret, vol, ret_max, ret_min]).shift(-n)

    return ret.T.swaplevel().T


def plot_change_scatter(s: pd.Series, df_mpd: pd.DataFrame):
    """
    Scatter plot for return / vol -> decrease / increase probability

    Parameters
    ----------
    s:  return / vol series to investigate
        Index: (ticker: str, date: DatetimeIndex)
    df_mpd: pd.DataFrame of market-based probability
        Columns: at least ["prDec", "prInc"]
        Index: (ticker: str, date: DatetimeIndex)
    """

    comp = pd.concat([
        df_mpd.set_index(["market", "idt"])[["prDec", "prInc"]],
        s], axis=1).sort_index().ffill()

    tickers = comp.index.get_level_values("ticker")
    color_map = dict(zip(colors.TABLEAU_COLORS[:len(tickers)], tickers))

    fig, ax = plt.subplots(2, 1, figsize=(12, 4))

    ax[0].axhline(y=0.2, c="red")
    ax[0].axhline(y=-0.2, c="red")
    [g.plot.scatter(
        x="prDec", y=s.name, ax=ax[0], s=10, label=name,
        color=color_map[name], alpha=0.5
    ) for name, g in comp.groupby("ticker")]

    ax[1].axhline(y=0.2, c="red")
    ax[1].axhline(y=-0.2, c="red")
    [g.plot.scatter(
        x="prInc", y=s.name, ax=ax[1], s=10, label=name,
        color=color_map[name], alpha=0.5
    ) for name, g in comp.groupby("ticker")]