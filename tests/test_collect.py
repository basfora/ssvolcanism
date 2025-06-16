"""Test data collection functions."""
from classes.collectdata import VolcanoData as vd
from classes.prediction import PredictionData as pred
from classes.basicfun import Basicfun as bf

import numpy as np
import datetime

def parameters_piton_periodI():
    """Parameters pre-defined for unit tests."""
    # Excel file >> Piton de la Fournaise
    name_file = 'PitondelaFournaise_data'
    # PITON period 1: 1 to 74 | period 2: 74 to 120
    # r1, rend = 1, 74
    p = 1

    qlong = 0.0024

    return name_file, p, qlong

def periodI_first5():
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.organize(p)
    # consider first 5
    mydates = edates[:5]
    myevol = evol[:5]
    mycvol = cvol[:6]
    return mydates, myevol, mycvol

def test_get_data():
    """Test data collection function."""
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.organize(p)

    # DATA IMPORT
    # size of the data
    assert len(edates) == 73
    assert len(evol) == 73
    assert len(cvol) == 74

    # check first 5 values
    assert evol[0] == 500000
    assert evol[1] == 12450000
    assert evol[2] == 4200000
    assert evol[3] == 37350000
    assert evol[4] == 1100000
    assert evol[:3] == [500000, 12450000, 4200000]
    #
    assert cvol[0] == 0
    assert cvol[1] == 500000
    assert cvol[2] == 12950000
    assert cvol[3] == 17150000
    assert cvol[4] == 54500000
    assert cvol[5] == 55600000
    #
    assert edates[0] == datetime.datetime(1936, 8, 1, 0, 0)
    assert edates[1] == datetime.datetime(1937, 8, 13, 0, 0)
    assert edates[2] == datetime.datetime(1938, 1, 1, 0, 0)
    assert edates[3] == datetime.datetime(1938, 12, 16, 0, 0)
    assert edates[4] == datetime.datetime(1942, 10, 1, 0, 0)



    th = 1e-4


def test_basic_stats():
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.organize(p)
    thv = 1e-1
    tht = 1e-4

    # consider first 5
    mydates = edates[:5]
    myevol = evol[:5]
    mycvol = cvol[:6]

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

    assert bf.get_limits(myevol) == (500000, 37350000)
    assert bf.get_total(myevol) == 55600000

    # volume functions
    assert bf.compute_delta_vol(mycvol[0], mycvol[1]) == 500000
    assert bf.compute_delta_vol(mycvol[0], mycvol[5]) == sum(myevol) == 55600000

    assert bf.m3_to_km3(55600000) == 0.0556
    assert bf.m3_to_km3([500000, 55600000]) == [0.0005, 0.0556]

    assert bf.compute_error(2, 1) == (1, 100.0)
    assert bf.compute_error(-1, 1) == (2, 200.0)
    assert bf.compute_error(0, 1) == (1, 100.0)
    assert bf.compute_error(8, 10) == (2, 20.0)


def test_time_functions():

    # first five values tested in previous test function (above)
    edates, evol, cvol = periodI_first5()

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
    bdates = [datetime.datetime(1992, 5, 30),
                    datetime.datetime(1996, 11, 26),
                    datetime.datetime(2003, 6, 1)]

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


def test_prediction_data():
    """Test prediction data collection."""
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.organize(p)

    # last eruption ID (real data, prediction will be ID + 1)
    last_eruption = 5
    # start prediction instance (PredictionData)
    mypred = pred(edates, evol, cvol, idx=last_eruption)

    # check initial values
    assert mypred.in_evol == evol[:5]
    assert mypred.in_cvol == cvol[:6]
    assert mypred.n == len(edates[:5])

    assert sum(mypred.dT_days) == mypred.time_total
    assert sum(mypred.in_evol) == mypred.cvol_delta


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





