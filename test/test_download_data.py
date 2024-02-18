import pandas as pd

from data_analysis.download_data import (get_fx, get_price, get_commodity,
                                         get_bond, get_inflation)


def test_get_price():
    df = get_price(tickers=["BAC", "C", "IYR", "SPY"],
                   start_date="2022-01-01",
                   end_date="2023-01-1")

    assert isinstance(df, pd.DataFrame)


def test_get_fx():
    df = get_fx(tickers=["GBP", "EUR", "JPY"],
                start_date="2022-01-01",
                end_date="2023-01-1")

    assert isinstance(df, pd.DataFrame)


def test_get_commodity():
    df = get_commodity(
        start_date='2022-01-12',
        end_date='2022-02-10'
    )

    assert isinstance(df, pd.DataFrame)


def test_get_inflation():
    df = get_inflation()

    assert isinstance(df, pd.DataFrame)


def test_get_bond():
    df = get_bond()

    assert isinstance(df, pd.DataFrame)
