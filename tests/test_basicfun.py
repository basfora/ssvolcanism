"""Test functions from BasicFun class."""
from classes.basicfun import Basicfun as bf

import numpy as np
import datetime


def test_basic_stats():
    """Test basic functions for statistics, from BasicFun class."""

    thv = 1e-1

    # consider first 5 from Piton de la Fournaise volcano
    myevol =  [500000, 12450000, 4200000, 37350000, 1100000]
    mycvol = [0, 500000, 12950000, 17150000, 54500000, 55600000]

    # test basic functions
    mean_evol, std_evol = bf.compute_mean_std(myevol)
    # MEAN OK
    assert abs(mean_evol - 11120000.00) < thv

    # sanity check - STD DEV
    stdev_pop = np.std(myevol, ddof=0)  # population std deviation
    stdev_sample = np.std(myevol, ddof=1)  # sample std deviation

    assert abs(stdev_pop - 13790235.6760137) < thv
    assert abs(stdev_sample - 15417952.1986547) < thv
    # in basic funcion, using population std deviation
    assert abs(std_evol - stdev_pop) < thv

    var_pop = bf.var_from_std(stdev_pop)
    assert abs(var_pop - 190170600000000) < thv
    assert abs(bf.std_from_var(var_pop) - stdev_pop) < thv

    assert bf.compute_limits(myevol) == (500000, 37350000)
    assert bf.get_total(myevol) == 55600000

    # volume functions
    assert bf.compute_delta_vol(mycvol[0], mycvol[1]) == 500000
    assert bf.compute_delta_vol(mycvol[0], mycvol[5]) == sum(myevol)
    assert bf.compute_delta_vol(mycvol[0], mycvol[5]) == 55600000

    assert bf.m3_to_km3(55600000) == 0.0556
    assert bf.m3_to_km3([500000, 55600000]) == [0.0005, 0.0556]

    assert bf.compute_error(1, 2) == (1, 100.0)
    assert bf.compute_error(1, -1) == (-2, 200.0)
    assert bf.compute_error(1, 0) == (-1, 100.0)
    assert bf.compute_error(10, 8) == (-2, 20.0)

    ans = bf.compute_error(1, [2, -1, 0])

    assert isinstance(ans[0], list)
    assert ans[0] == [1, -2, -1]
    assert isinstance(ans[1], list)
    assert ans[1] == [100, 200, 100]


def test_time_functions():
    """Test basic functions for time, from BasicFun class."""

    edates = [datetime.date(1936, 8, 1), datetime.date(1937, 8, 13),
              datetime.date(1938, 1, 1), datetime.date(1938, 12, 16),
              datetime.date(1942, 10, 1)]
    # --------------------------
    mydays = 377
    myyears = 2

    assert bf.years_to_days(myyears) == 730.5
    assert bf.days_to_years(mydays) == 1.0322

    # first five intervals (days)
    dT_first5 = [377, 141, 349, 1385]
    # dT_days
    eintervals = bf.compute_intervals(edates)

    assert bf.compute_intervals([1]) == []
    assert len(eintervals) == len(edates) - 1 == 4
    assert eintervals[0] == 377
    assert eintervals[1] == 141
    assert eintervals[2] == 349
    assert eintervals[3] == 1385

    # --
    bdates = [datetime.date(1992, 5, 30),
                    datetime.date(1996, 11, 26),
                    datetime.date(2003, 6, 1)]

    bintervals = bf.compute_intervals(bdates)

    assert len(bintervals) == len(bdates) - 1 == 2
    assert bintervals[0] == 1641 # B-C
    assert bintervals[1] == 2378 # C-M
    assert bf.compute_days(bdates[0], bdates[-1]) == 4019 # B-M

    # timeline of eruptions
    etimeline = bf.compute_timeline(eintervals)
    btimeline = bf.compute_timeline(bintervals)

    assert len(etimeline) == len(edates) == len(eintervals) + 1 == 5
    assert etimeline[0] == 0
    assert etimeline[1] == 377
    assert etimeline[2] == 518
    assert etimeline[3] == 867
    assert etimeline[4] == 2252
    assert sum(eintervals) == etimeline[-1] == 2252
    #-
    assert len(btimeline) == len(bdates) == len(bintervals) + 1 == 3
    assert btimeline[0] == 0
    assert btimeline[1] == 1641
    assert btimeline[2] == 4019
    assert sum(bintervals) == btimeline[-1] == 4019


def test_rate_functions():

    # research values (original)
    qlong = bf.Qy_to_Qday(0.0024)
    qperiod1 = bf.Qy_to_Qday(0.0107)
    qperiod2 = bf.Qy_to_Qday(0.0228)

    assert qlong ==  0.0024 * 1e9 / 365.25  # m3/day
    assert qperiod1 == 0.0107 * 1e9 / 365.25  == 29295.003422313483 # m3/day
    assert qperiod2 == 0.0228 * 1e9 / 365.25  # m3/day

    assert round(bf.Qday_to_Qy(qlong), 4) == 0.0024
    assert round(bf.Qday_to_Qy(qperiod1), 4) == 0.0107
    assert round(bf.Qday_to_Qy(qperiod2),4) == 0.0228

    CV0, CVf = 0, 658060000
    dT_days = 22502
    q_hand = 29244.51159896898 # 29295.003422313483 (transformed to m3/day from 0.0107 km3/yr)

    # computation of rate - OK (qhand matches Q research for period 1)
    assert round(bf.compute_q(CV0, CVf, dT_days), 4) == round(q_hand, 4) == 29244.5116
    assert round(bf.Qday_to_Qy(29244.5116), 4) == 0.0107


def test_dimension():
    # sanity check for dimensions of the data
    CV1 = 55600000
    N = 100
    dT_days = [377, 141, 349, 1385]
    qhat = 24689

    CV1_array = [CV1] * N
    # sampling time intervals
    dTsim = np.random.choice(dT_days, N, replace=True)

    CV2_simple = CV1 + qhat * dTsim
    CV2 = CV1_array + qhat * dTsim

    assert len(CV2_simple) == N
    assert len(CV2) == N
    assert CV2.all() == CV2_simple.all()





