from data_analysis.explore_data import calc_log_ret
from data_analysis.download_data import get_price


def test_calc_log_ret():
    df = get_price(tickers=["BAC", "C", "IYR", "SPY"],
                   start_date="2022-01-01", end_date="2023-12-31",
                   use_cache=True)
    ret = calc_log_ret(df)

    assert False