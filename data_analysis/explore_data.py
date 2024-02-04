import pandas as pd
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from typing import Callable, Any

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


def calc_rolling_stat(df: pd.DataFrame, month: int = 6) -> pd.DataFrame:
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

    def ret(x): return np.log(x.iloc[-1] / x.iloc[0])

    def vol(x): return x.iloc[:-1].std()

    def ret_max(x): return np.log(x.max() / x.iloc[0])

    def ret_min(x): return np.log(x.min() / x.iloc[0])

    n = month * 20 + 1
    ret = df.rolling(f"{n}D").aggregate([ret, vol, ret_max, ret_min]).shift(-n)

    return ret.stack(level=0).swaplevel()


def rename_stat_df(df_stat: pd.DataFrame,
                   name_map: dict[str, str]) -> pd.DataFrame:
    """
    Rename stat DataFrame [ticker] -> probability dataframe [market]

    Parameters
    ----------
    df_stat: pd.DataFrame
        Index: (ticker, date)
        Columns: ret, vol, ret_min, ret_max
    name_map: dict
        dictionary of {ticker: market}, e.g. {"sp6m": "SPY", "sp12m": "SPY"}

    Returns
    -------
    pd.DataFrame:
        Index: (market, date)
        Columns: ret, vol, ret_min, ret_max
    """

    df_list = []
    keys = []
    for k, v in name_map.items():
        df_list.append(df_stat.loc[v])
        keys.append(k)

    return pd.concat(df_list, axis=0, keys=keys).unstack(level=0)


def merge_prob_stat(df_stat: pd.DataFrame,
                    df_mpd: pd.DataFrame) -> pd.DataFrame:
    """
    Combine probability and return data

    Parameters
    ----------
    df_stat: pd.DataFrame of rolling stat calculated
    df_mpd: pd.DataFrame of bi-weekly MPD data

    Returns
    -------
    pd.DataFrame:
        Columns:
            prDec: probability of large decrease
            prInc: probability of large increase
            ret: next X-month return
            vol: next X-month close-to-close volatility
            ret_min: next X-month maximum drawdown
            ret_max: next X_month maximum return
    """
    comp = pd.concat([
        df_mpd.set_index(["market", "idt"])[["prDec", "prInc"]],
        df_stat.stack().swaplevel()], axis=1).sort_index()
    comp.index.names = ("market", "idt")
    comp.loc[:, df_stat.stack().columns] = comp.groupby(
        "market")[df_stat.stack().columns].ffill()
    comp = comp.dropna(how="any")

    return comp


def eval_large_change_prob(df: pd.DataFrame,
                           ub: float = 0.2,
                           lb: float = -0.2) -> pd.DataFrame:
    """
    Evaluate average probability for returns with large changes

    Parameters
    ----------
   df: pd.DataFrame
        Columns: at least ["prDec", "prInc", "ret"]
        Index: (market: str, date: DatetimeIndex)
    ub: upside large change threshold, e.g. 0.2 for equity
    lb: downside large change threshold, e.g. -0.2 for equity

    Returns
    -------
    pd.DataFrame
    """
    markets = df.index.get_level_values("market").unique()
    results = {}

    for m in markets:
        groupby = np.select(
            [df.loc[m, "ret"] > ub, df.loc[m, "ret"] < lb],
            [1, -1],
            default=0
        )
        results[m] = df.loc[m][["prDec", "prInc"]].groupby(
            groupby).mean().unstack()

    return pd.DataFrame(results).T


def hurst_exponent(x, max_lag=20):
    """Returns the Hurst Exponent of the time series"""

    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [x.diff(lag).std() for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def plot_change_scatter(df: pd.DataFrame, col: str,
                        hline_levels: list[float] = None):
    """
    Scatter plot for return / vol -> decrease / increase probability

    Parameters
    ----------
    df: pd.DataFrame
        Columns: at least ["prDec", "prInc", col]
        Index: (market: str, date: DatetimeIndex)
    col: str
        Column name of prDec / prInc -> series plot
    """

    tickers = df.index.get_level_values("market").unique()
    color_map = dict(zip(
        tickers, list(colors.TABLEAU_COLORS.keys())[:len(tickers)]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter Plot
    [g.plot.scatter(
        x="prDec", y=col, ax=ax[0], s=10, label=name,
        color=color_map[name], alpha=0.5
    ) for name, g in df.groupby("market")]

    [g.plot.scatter(
        x="prInc", y=col, ax=ax[1], s=10, label=name,
        color=color_map[name], alpha=0.5
    ) for name, g in df.groupby("market")]

    if hline_levels is not None:
        for i in hline_levels:
            ax[0].axhline(y=i, c="red")
            ax[1].axhline(y=i, c="red")

    plt.show()


def prep_regression_stat(df: pd.DataFrame,
                         resample_week: int = 2,
                         save_fig_name: str = None):
    """
    Calculate ACF / PACF / Seasonal test to prepare for regression

    I remove ADFULLER test because it's very similar with ACF / PACF, and
    it will focus on 1 period AR of overlapping periods.

    Parameters
    ----------
    df: pd.DataFrame
        Columns: at least ["prDec", "prInc", "ret"]
        Index: (market: str, date: DatetimeIndex)
    resample_week: int
        Resample return series for evenly distributed interval
    save_fig_name: str
        If None, show() plot directly otherwise save with prefix defined here.
    """
    markets = df.index.get_level_values("market").unique()

    figsize = (12, 4 * len(markets))
    fig, ax = plt.subplots(len(markets), 2, figsize=figsize)
    fig_s, ax_s = plt.subplots(len(markets), 1, figsize=figsize)
    fig_f, ax_f = plt.subplots(len(markets), 1, figsize=figsize)

    n = 0
    dominant_freq = {}
    season_dict = {}
    for m in markets:
        s = df.loc[m, "ret"].resample(f"{resample_week}W").first().ffill()

        # ACF / PACF
        sm.graphics.tsa.plot_acf(s, ax=ax[(n, 0)], title="ACF")
        sm.graphics.tsa.plot_pacf(s, ax=ax[(n, 1)], title="PACF")
        ax[(n, 0)].set_ylabel(m)

        # Seasonal Decompose
        season_results = seasonal_decompose(s)
        season_dict[m] = pd.DataFrame({"trend": season_results.trend,
                                       "seasonal": season_results.seasonal,
                                       "resid": season_results.resid,
                                       "observed": season_results.observed})
        season_dict[m].plot(ax=ax_s[n], ylabel=m)

        # Season FFT
        fft_result = np.fft.fft(s)
        power_spectrum = np.abs(fft_result) ** 2
        freq = np.fft.fftfreq(len(s), d=resample_week)
        dominant_freq[m] = freq[np.argsort(np.abs(fft_result))[::-1][1:]]
        dominant_freq[m] = 1 / dominant_freq[m][dominant_freq[m] > 0][0]
        ax_f[n].stem(1 / freq, power_spectrum)
        ax_f[n].set_ylabel(m)
        ax_f[n].set_title(f"max = {dominant_freq[m]}")
        ax_f[n].set_xlim(0, s.shape[0])

        n += 1

    fig.suptitle("ACF vs PACF")
    fig_s.suptitle("Seasonal Decompose")
    fig_f.suptitle("FFT Power Spectrum (xlabel = weeks)")

    if save_fig_name is None:
        fig.show()
        fig_s.show()
        fig_f.show()
    else:
        fig.savefig(f"{save_fig_name}_acf.png")
        fig_s.savefig(f"{save_fig_name}_seasonal.png")
        fig_f.savefig(f"{save_fig_name}_fft.png")

    return dominant_freq


def compare_regressions(df: pd.DataFrame,
                        model: Callable,
                        model_ret_name: str = "endog",
                        model_prob_name: str = "exog",
                        resample_week: int = 2) -> dict[str, Any]:
    """
    Compare regression results of model with and without MPD data on returns

    Parameters
    ----------
    df: pd.DataFrame
        Columns: at least ["prDec", "prInc", "ret"]
        Index: (market: str, date: DatetimeIndex)
    resample_week: int
        Resample interval to convert return data
    model: Callable
        Partial function to initialize regression class,
        e.g. SARIMAX, GARCHX, Markov Switching
    model_ret_name: str
        parameter name of the input return series
    model_prob_name: str
        parameter name of the input probability series
    model: str
        Partial function of model to use

    Returns
    -------
    dictionary of {(market, self/prob): SARIMAResult class instance}
    """
    markets = df.index.get_level_values("market").unique()
    results = {}
    for m in markets:
        df_resample = df.loc[m].resample(f"{resample_week}W").first().ffill()
        y = df_resample["ret"]
        X = df_resample[["prDec", "prInc"]]

        results[(m, "without")] = model(**{model_ret_name: y}).fit()
        results[(m, "with")] = model(**{model_ret_name: y,
                                        model_prob_name: X}).fit()

    return results


def compare_regression_eval(results: dict[str, Any],
                            metrics: str) -> pd.DataFrame:
    """
    Compare regression results by certain metrics

    Parameters
    ----------
    results: dict[str, Any]
        results of compare_regressions()
    metrics: str
        evaluation metrics to compare

    Returns
    -------
    pd.DataFrame:
        Columns: with (MPD data), without (MPD data), diff (= without - with)
        Index: markets
    """
    df = pd.Series({
        k: getattr(v, metrics) if hasattr(v, metrics) else np.nan
        for k, v in results.items()
    }).unstack()

    df.loc[:, "diff"] = df["with"] - df["without"]
    return df

