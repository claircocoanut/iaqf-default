from datetime import datetime

import nasdaqdatalink
import pandas as pd
import quandl

myAPIkey = "DKEpSA2RKyvtpZKmVWGv"
nasdaqdatalink.ApiConfig.api_key = myAPIkey


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
    df = nasdaqdatalink.get_table(
        'QUOTEMEDIA/PRICES',
        date=','.join(dates.tolist()),
        ticker=','.join(tickers),
        paginate=True
    )
    df = df.set_index(["date", "ticker"])["adj_close"].unstack()

    return df


@cache_df(name="spy")
def get_spy(start_date: str | datetime,
            end_date: str | datetime) -> pd.DataFrame:
    """
    Download SPY price for beta calculation

    Parameters
    ----------
    start_date: get price data from start_date to end_date (both inclusive)
    end_date: get price data from start_date to end_date (both inclusive)

    Returns
    -------
    pd.DataFrame of required price data for analysis
    """

    return get_price(use_cache=False,
                     tickers=["SPY"],
                     start_date=start_date,
                     end_date=end_date)["SPY"]


@cache_df(name="fx_price")
def get_fx(tickers: list[str],
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
    df = nasdaqdatalink.get_table(
        'EDI/CUR',
        date=','.join(dates.tolist()),
        code=','.join(tickers),
        paginate=True
    )
    df = df.set_index(["date", "code"])["rate"].unstack()

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


@cache_df(name="commodity_price")
def get_commodity(
        future_name: list[str] = ['CMX_SI_SI', 'CBT_C_C', 'NYX_EMA_EMA', 'CBT_S_S', 'CMX_GC_GC', 'CBT_W_W'],
        commodity: list[str] = ['silver', 'corn1', 'corn2', 'soybean', 'gold', 'wheat'],
        start_date: str | datetime = '2006-01-12',
        end_date: str | datetime = '2024-01-10'
):
    data_list = list()
    for i, name in enumerate(future_name):
        df = quandl.get('OWF/' + name + '_6M' + '_IVM', returns='pandas', start_date=start_date, end_date=end_date,
                        api_key=myAPIkey)
        df = df[['Future', 'DtT']]
        df.rename(columns={'Future': 'FuturePrice', 'DtT': 'DtT(expiration)'}, inplace=True)
        df = df[(df['DtT(expiration)'] >= 180.0) & (df['DtT(expiration)'] <= 210.0)]
        df.ffill(inplace=True)
        assert df.isna().any().any(), 'Please clean data'
        df.drop(columns=['DtT(expiration)'], inplace=True)
        df.rename(columns={'FuturePrice': commodity[i]}, inplace=True)
        data_list.append(df)

    df = pd.concat(data_list, axis=1).dropna()
    return df


def get_inflation():
    inf1yr = pd.read_csv("../data/EXPINF1YR.csv", index_col='DATE')
    inf2yr = pd.read_csv('../data/EXPINF2YR.csv', index_col='DATE')
    inf5yr = pd.read_csv('../data/EXPINF5YR.csv', index_col='DATE')
    inflation = inf1yr.join([inf2yr, inf5yr], how='outer')
    inflation.index = pd.to_datetime(inflation.index)
    return inflation


@cache_df("bond_price.pkl")
def get_bond(coupon_rate: float = 0.02):
    treasury = quandl.get('USTREASURY/REALYIELD')
    treasury.index = pd.to_datetime(treasury.index)

    def bond_price(yield_rate, maturity_years, face_value=100, coupon_rate=coupon_rate):
        period_rate = yield_rate / 2
        total_periods = maturity_years * 2
        coupon_payment = face_value * (coupon_rate / 2)
        price = sum(coupon_payment / (1 + period_rate) ** (i + 1) for i in range(total_periods)) + face_value / (
                    1 + period_rate) ** total_periods
        return price

    treasury['5YR_Price'] = treasury['5 YR'].apply(lambda y: bond_price(y / 100, 5))
    treasury['10YR_Price'] = treasury['10 YR'].apply(lambda y: bond_price(y / 100, 10))
    treasury.index = pd.to_datetime(treasury.index)

    return treasury
