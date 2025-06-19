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

    qperiod = 0.0024

    return name_file, p, qperiod

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
    assert edates[0] == datetime.date(1936, 8, 1)
    assert edates[1] == datetime.date(1937, 8, 13)
    assert edates[2] == datetime.date(1938, 1, 1)
    assert edates[3] == datetime.date(1938, 12, 16)
    assert edates[4] == datetime.date(1942, 10, 1)



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


def test_prediction_input():
    """Test data collection to prediction creation."""
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.organize(p)

    # last eruption ID (real data, prediction will be ID + 1)
    last_eruption = 5
    # start prediction instance (PredictionData)
    pp = pred(edates, evol, cvol, id_last=last_eruption)

    # check initial values
    assert pp.in_evol == evol[:5]
    assert pp.in_cvol == cvol[:6]
    assert pp.n == len(edates[:5])

    # check OneEruption instance
    assert pp.oe.id == last_eruption + 1 == 6 # prediction ID
    # T1 - initial values saved correctly (eruption 5)
    assert pp.oe.evol.t1 == pp.in_evol[-1] == 1100000  # eruption volume at T1 (last eruption)
    assert pp.oe.cvol.t1 == pp.in_cvol[-1] == 55600000  # cumulative volume at T1 (last eruption)
    assert pp.oe.dT.t1 == 2252 # time in days from 1936-08-01
    assert pp.oe.date.t1 == pp.in_edates[-1] == datetime.date(1942, 10, 1)  # date of the last eruption (T1)

    # T2 - target values (eruption 6 real data), idx = 5 in lists
    assert pp.oe.date.real == edates[5] == datetime.date(1943, 2, 1)
    assert pp.oe.evol.real == evol[5] == 500000  # eruption volume at T2
    assert pp.oe.cvol.real == cvol[6] == 56100000  # cumulative volume at T2
    assert pp.oe.dT.real == 123 # time in days from 1942-10-01 to 1943-02-01

def test_historical_stats():
    """Test stats computation for historical data in prediction."""
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.organize(p)

    # last eruption ID (real data, prediction will be ID + 1)
    last_eruption = 5
    # start prediction instance (PredictionData)
    pp = pred(edates, evol, cvol, id_last=last_eruption)

    # checking values
    mydates, myevol, mycvol = periodI_first5()
    evol_mean, evol_std = np.mean(myevol), np.std(myevol, ddof=0)


    # CVOL
    assert pp.cvol_delta == bf.compute_delta_vol(mycvol[0], mycvol[-1]) == 55600000
    assert pp.cvol_delta == pp.evol_sum == 55600000
    assert sum(pp.in_evol) == pp.cvol_delta

    # EVOL
    thv = 1e-1
    assert abs(pp.evol_mean - 11120000) <= thv
    assert abs(pp.evol_mean - evol_mean) <= thv
    assert abs(pp.evol_std - evol_std) <= thv
    assert abs(pp.evol_std - 13790235.6760137) <= thv

    # DAYS / TIME
    assert sum(pp.dT_days) == pp.time_total == 2252
    # intervals
    assert len(pp.dT_days) == pp.n - 1 == 4
    assert pp.dT_days[0] == 377
    assert pp.dT_days[1] == 141
    assert pp.dT_days[2] == 349
    assert pp.dT_days[3] == 1385
    # timeline
    assert pp.timeline[0] == 0
    assert pp.timeline[1] == 377
    assert pp.timeline[2] == pp.dT_days[0] + pp.dT_days[1] == pp.timeline[1] + pp.dT_days[1]== 518

    assert pp.timeline[3] == pp.dT_days[0] + pp.dT_days[1] + pp.dT_days[2] == 867
    assert pp.timeline[3] == pp.timeline[2]  + pp.dT_days[2] == 867

    assert pp.timeline[4] == pp.dT_days[0] + pp.dT_days[1] + pp.dT_days[2] + pp.dT_days[3] == 2252
    assert pp.timeline[-1] == pp.timeline[3]  + pp.dT_days[3] == pp.time_total == 2252


def test_deterministic_prediction():
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.organize(p)

    # - check Q period
    qperiod = piton_data.Q1
    qperiod1 = bf.Qy_to_Qday(0.0107)
    # saved the right value
    assert qperiod == qperiod1
    # which is
    assert round(qperiod, 4) == 29295.0034

    qperiod_yr = bf.Qday_to_Qy(qperiod)

    assert qperiod == bf.Qy_to_Qday(0.0107)
    assert qperiod_yr == 0.0107

    # last eruption ID (real data, prediction will be ID + 1)
    last_eruption = 5
    # start prediction instance (PredictionData)
    pp = pred(edates, evol, cvol, id_last=last_eruption)
    pp.set_qperiod(qperiod)



    # RUN deterministic method
    pp.deterministic_method()
    # check results

    assert pp.oe.id == last_eruption + 1 == 6  # prediction ID
    assert pp.oe.date.real == edates[5] == datetime.date(1943, 2, 1)  # date of the last eruption (T2)

    # by hand
    CV1 = 55600000
    dT = 123  # days
    q = 29295.0034

    CV2 = CV1 + q * dT  # cumulative volume at T2

    assert abs(CV1 - pp.oe.cvol.t1) < 1
    assert dT == pp.oe.dT.real == pp.oe.dT.real == 123
    assert abs(pp.oe.q_period - q) < 0.1


    # hand and state equation OK
    assert round(CV2, 4) == 59203285.4182
    assert round(CV2, 4) == round(bf.state_equation(CV1, q, dT), 4)
    # hand and saved prediction
    assert round(pp.oe.cvol.det.value, 4) == round(bf.state_equation(pp.oe.cvol.t1, pp.oe.q_period, pp.oe.dT.real), 4)
    assert round(CV2) == round(pp.oe.cvol.det.value)


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





