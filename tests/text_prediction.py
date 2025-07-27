"""Test functions from Prediction class."""
import datetime
import numpy as np

from classes.basicfun import Basicfun as bf
from classes.collectdata import VolcanoData as vd
from classes.prediction import PredictionData as pred
# from classes.subset import MySubset as ms

def parameters_piton_periodI():
    """Parameters pre-defined for unit tests."""
    # Excel file >> Piton de la Fournaise
    name_file = 'TablePiton'
    # PITON period 1: 1 to 74 | period 2: 74 to 120
    # r1, rend = 1, 74
    p = 1

    qperiod = 0.0107

    return name_file, p, qperiod

def periodI_first5():
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.define_periods(p)
    # consider first 5
    mydates = edates[:5]
    myevol = evol[:5]
    mycvol = cvol[:6]
    return mydates, myevol, mycvol


def test_prediction_input():
    """Test data collection to prediction creation."""
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edateall, evolall, cvolall = piton_data.define_periods(p)

    # last eruption ID (real data, prediction will be ID + 1)
    last_eruption = 5
    edates, evol, cvol = piton_data.output_real_data(0, last_eruption)

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

    enext, evolnext, cvolnext = piton_data.output_next(last_eruption)
    pp.save_real_next(enext, evolnext, cvolnext)

    # T2 - target values (eruption 6 real data), idx = 5 in lists
    assert pp.oe.date.t2 == edateall[5] == datetime.date(1943, 2, 1)
    assert pp.oe.evol.t2 == evolall[5] == 500000  # eruption volume at T2
    assert pp.oe.cvol.t2 == cvolall[6] == 56100000  # cumulative volume at T2
    assert pp.oe.dT.t2 == 123 # time in days from 1942-10-01 to 1943-02-01

def test_historical_stats():
    """Test stats computation for historical data in prediction."""
    name_file, p, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edatesall, evolall, cvolall = piton_data.define_periods(p)

    # last eruption ID (real data, prediction will be ID + 1)
    last_eruption = 5
    # get relevant data from the file
    edates, evol, cvol = piton_data.output_real_data(0, last_eruption)

    # start prediction instance (PredictionData)
    pp = pred(edates, evol, cvol, last_eruption)
    pp.set_period_info(piton_data.periods[2].q)

    enext, evolnext, cvolnext = piton_data.output_next(last_eruption)
    pp.save_real_next(enext, evolnext, cvolnext)

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
    edatesall, evolall, cvolall = piton_data.define_periods(p)

    # - check Q period
    qperiod = piton_data.periods[1].q
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

    # get relevant data from the file
    edates, evol, cvol = piton_data.output_real_data(0, last_eruption)

    # start prediction instance (PredictionData)
    pp = pred(edates, evol, cvol, id_last=last_eruption)
    pp.set_period_info(qperiod)

    enext, evolnext, cvolnext = piton_data.output_next(last_eruption)
    pp.save_real_next(enext, evolnext, cvolnext)

    # RUN deterministic method
    pp.deterministic_method()
    # check results

    assert pp.oe.id == last_eruption + 1 == 6  # prediction ID
    assert pp.oe.date.t2 == edatesall[5] == datetime.date(1943, 2, 1)  # date of the last eruption (T2)

    # by hand
    CV1 = 55600000
    dT = 123  # days
    q = 29295.0034

    CV2 = CV1 + q * dT  # cumulative volume at T2

    assert abs(CV1 - pp.oe.cvol.t1) < 1
    assert dT == pp.oe.dT.t2 == pp.oe.dT.t2 == 123
    assert abs(pp.oe.qperiod - q) < 0.1


    # hand and state equation OK
    assert round(CV2, 4) == 59203285.4182
    assert round(CV2, 4) == round(bf.state_equation(CV1, q, dT), 4)
    # hand and saved prediction
    assert round(pp.oe.cvol.det.value, 4) == round(bf.state_equation(pp.oe.cvol.t1, pp.oe.qperiod, pp.oe.dT.t2), 4)
    assert round(CV2) == round(pp.oe.cvol.det.value)
