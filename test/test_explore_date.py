import pandas as pd
from pytest import fixture

from data_analysis.explore_data import (
    calc_rolling_stat,
    merge_prob_stat,
    rename_stat_df,
    eval_large_change_prob,
    plot_change_scatter,
    prep_regression_stat,
    compare_regressions,
    compare_regression_eval
)
from data_analysis.download_data import get_price, get_mpd


@fixture
def test_rolling_stat():
    df = get_price(tickers=["BAC", "C", "IYR", "SPY"],
                   start_date="2022-01-01", end_date="2023-12-31",
                   use_cache=True)
    stat = calc_rolling_stat(df)
    stat_rename = rename_stat_df(
        stat, dict(zip(["bac", "citi", "iyr", "sp6m", "sp12m"],
                       ["BAC", "C", "IYR", "SPY", "SPY"])))

    return stat_rename


@fixture
def test_df(test_rolling_stat):
    mpd = get_mpd()
    df = merge_prob_stat(test_rolling_stat, mpd)

    return df


def test_eval_large_change_prob(test_df):
    eval_large_change_prob(test_df)


def test_plot_change_scatter(test_df):
    plot_change_scatter(df=test_df, col="ret",
                        hline_levels=[-0.2, 0.2])


def test_prep_regression_stat(test_df):
    prep_regression_stat(test_df)


def test_compare_regressions(test_df):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from arch import arch_model
    import statsmodels.api as sm
    from functools import partial

    arima = compare_regressions(test_df, partial(SARIMAX, order=(1, 0, 1)))
    garch = compare_regressions(
        test_df, partial(arch_model, p=1, q=1, mean='ARX', vol='GARCH'),
        model_ret_name="y", model_prob_name="x",
    )

    markov = compare_regressions(
        test_df, partial(sm.tsa.MarkovRegression, k_regimes=3))

    markov_vol = compare_regressions(
        test_df, partial(sm.tsa.MarkovRegression, k_regimes=3,
                         switching_variance=True))
