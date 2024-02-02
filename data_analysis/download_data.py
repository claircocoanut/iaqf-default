from datetime import datetime
from typing import Callable
from functools import wraps

import nasdaqdatalink
import pandas as pd

nasdaqdatalink.ApiConfig.api_key = "DKEpSA2RKyvtpZKmVWGv"


def cache_df(name: str):
    """
    Wrapper function to use locally stored data
    """
    def inner_func(func):
        def wrapper(use_cache: bool = False, *args, **kwargs):
            """
            use_cache: if True, try read local pickle file
            """
            try:
                assert use_cache
                df = pd.read_pickle(f"../data/{name}.pkl")
            except (AssertionError, FileNotFoundError):
                df = func(*args, **kwargs)
                df.to_pickle(f"../data/{name}.pkl")
            return df
        return wrapper

    return inner_func


@cache_df(name="equity_price")
def get_price(tickers: list[str],
              start_date: str | datetime,
              end_date: str | datetime) -> pd.DataFrame:
    """
    Download required data from Quandl QUOTEMEDIA/PRICES and
    save to pickle file for future rerun.

    Parameters
    ----------
    tickers: download price data of ticker from
    start_date: get price data from start_date to end_date (both inclusive)
    end_date: get price data from start_date to end_date (both inclusive)

    Returns
    -------
    pd.DataFrame of required price data for analysis
    """
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")
    df = nasdaqdatalink.get_table('QUOTEMEDIA/PRICES',
                                  date=','.join(dates.tolist()),
                                  ticker=','.join(tickers),
                                  paginate=True)
    df = df.set_index(["date", "ticker"])["adj_close"].unstack()

    return df


def get_mpd() -> pd.DataFrame:
    """
    Get Market-Based Probability data from local csv

    Returns
    -------
    pd.DataFrame of downloaded data with asset type
        Columns:
            type: asset type, e.g. equity, commodity ....
            Please refer to column_def.csv for other columns' description;
        Index:
            DatetimeIndex in bi-weekly frequency
    """
    data = pd.read_csv("../data/mpd_stats.csv")
    ticker_def = pd.read_csv("../data/ticker_def.csv",
                             index_col=["ticker"])

    data = data.merge(ticker_def, left_on=["market"],
                      right_index=True, how="left")
    data.loc[:, "idt"] = pd.to_datetime(data["idt"])

    return data
