import numpy as np
import pandas as pd

from .download_data import get_spy


def downside_beta(ret: pd.Series, mkt_ret: pd.Series,
                  ann_factor: int = 252) -> float:
    """
    Calculate downside beta = correlation of strategy return and market
    return when market return < 0

    Parameters
    ----------
    ret: pd.Series of strategy return;
    mkt_ret: pd.Series of market return, same interval as ret;
    ann_factor: annualizing factor.

    Returns
    -------
    downside beta value
    """
    idx = ret.index.intersection(mkt_ret.loc[mkt_ret < 0].index)
    return ret.loc[idx].corr(mkt_ret.loc[idx])


def sharpe_ratio(ret: pd.Series, rf_ret: pd.Series = None,
                 ann_factor: int = 252) -> float:
    """
    Calculate sharpe ratio

    Parameters
    ----------
    ret: pd.Series of strategy return;
    rf_ret: pd.Series of risk-free rate/return, same interval as ret;
    ann_factor: annualizing factor.

    Returns
    -------
    sharpe ratio
    """
    if rf_ret is not None:
        ret = ret - rf_ret
    return ret.mean() / ret.std() * np.sqrt(ann_factor)


def sortino_ratio(ret: pd.Series, rf_ret: pd.Series = None,
                  ann_factor: int = 252) -> float:
    """
    Calculate sortino ratio

    Parameters
    ----------
    ret: pd.Series of strategy return;
    rf_ret: pd.Series of risk-free rate/return, same interval as ret;
    ann_factor: annualizing factor.

    Returns
    -------
    sortino ratio
    """
    if rf_ret is not None:
        ret = ret - rf_ret

    return ret.mean() / ret.loc[ret < 0].std() * np.sqrt(ann_factor)


def var(s: pd.Series, ann_factor: int, q: float = 0.05) -> float:
    """
    Calculate VaR = n-th quantile

    Parameters:
        s (pd.Series):
            Return series of certain asset;
        ann_factor (float):
            Annualization factor to match return and Fama-French data;
        q (float):
            quantile used to calculate VaR.

    Returns:
        VaR value of input asset returns
    """
    return np.quantile(s, q=q) * np.sqrt(ann_factor)


def cvar(s: pd.Series, ann_factor: int, q: float = 0.05) -> float:
    """
    Calculate the mean of the returns at or below the q quantile

    Parameters:
        s (pd.Series):
            Return series of certain asset;
        ann_factor (float):
            Annualization factor to match return and Fama-French data;
        q (float):
            Quantile used to calculate CVaR.

    Returns:
        CVaR value of input asset returns
    """
    return s.loc[s < np.quantile(s, q=q)].mean() * np.sqrt(ann_factor)


# %%
def max_drawdown(s: pd.Series, return_dict: bool = False) -> float | pd.Series:
    """
    Calculate the maximum drawdown, peak date, trough date, and recovery date

    Parameters:
        s (pd.Series):
            Return series of certain asset

    Returns:
        pd.Series of all statistics of given asset
    """
    s_cum = (s + 1).cumprod()
    s_cum_max = s_cum.cummax()
    pct_to_peak = s_cum / s_cum_max - 1
    drawdown = min(pct_to_peak)

    if return_dict:
        trough_date = pct_to_peak[pct_to_peak == drawdown].index[0]
        peak_cum = s_cum_max[pct_to_peak == drawdown][0]
        peak_date = s_cum[s_cum == peak_cum].index[0]
        is_recovered = ((s_cum.index > trough_date) &
                        (s_cum >= peak_cum))
        recovery_date = s_cum.loc[is_recovered].index[0] \
            if any(is_recovered) else None

        return pd.Series({
            "drawdown": drawdown,
            "trough_date": trough_date,
            "peak_date": peak_date,
            "recovery_date": recovery_date
        })
    else:
        return drawdown


def eval_return(cum_ret: pd.Series,
                resample_interval: str = None,
                ann_factor: float = 26):
    """
    Calculate metrics to compare the results with different parameters:
    1. cumulative total return
    2. Sharpe ratio (with risk-free rate / market return as benchmark)
    3. Sortino ratio
    4. Beta for 3 Fama-French factors
    5. Tail - VaR
    6. Tail - CVaR
    7. Tail - Max Drawdown
    8. Risk - Downside beta

    Parameters
    ----------
    cum_ret: pd.Series
        cumulative returns;
    resample_interval: str
        Default = 2W, i.e. resample to two-week interval for sharpe calculation
    ann_factor: int
        annualization factor to convert into annualized sharpe ratio

    Returns
    -------
    dict of metrics
    """
    if resample_interval is not None:
        cum_ret = cum_ret.resample(resample_interval).ffill()
    ret = cum_ret.diff().dropna()

    spy = get_spy(start_date=cum_ret.index.min(), end_date=cum_ret.index.max())
    spy = spy.resample("D").ffill().reindex(ret.index).pct_change()

    return {
        "total_return": cum_ret.iloc[-1],
        "mean": ret.mean(),
        "std": ret.std(),
        "skew": ret.skew(),
        "kurtosis": ret.kurt(),
        "sharpe": sharpe_ratio(ret, ann_factor=ann_factor),
        "sharpe_mkt": sharpe_ratio(ret, spy, ann_factor),
        "sortino": sortino_ratio(ret, ann_factor=ann_factor),
        "var": var(ret, ann_factor),
        "cvar": cvar(ret, ann_factor),
        "max_drawdown": max_drawdown(ret),
        "downside_beta": downside_beta(ret, spy),
    }
