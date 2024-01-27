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
                df = pd.read_pickle(f"{name}.pkl")
            except (AssertionError, FileNotFoundError):
                df = func(*args, **kwargs)
                df.to_pickle(f"{name}.pkl")
            return df
        return wrapper

    return inner_func


@cache_df(name="equity_price")
def download_price_data(tickers: list[str],
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
                                  ticker=','.join(tickers))
    df = df.set_index(["date", "ticker"])[
        ["adj_close", "volume"]].unstack()
    return df
