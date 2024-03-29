from datetime import datetime

from numpy.random import normal
import pandas as pd
import numpy as np

from data_analysis.analyse_results import eval_return


def test_eval_return():

    cum_ret = pd.Series(
        normal(0.001, 0.3, 200),
        pd.date_range(end=datetime.today(), freq="1W", periods=200)
    ).cumsum()

    results = eval_return(cum_ret, resample_interval="2W", ann_factor=52)

    assert isinstance(results, dict)
    assert [not np.isnan(x) for x in results.values()]
