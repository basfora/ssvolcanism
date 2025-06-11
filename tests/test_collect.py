"""Test data collection functions."""
from classes.collectdata import VolcanoData as vd
from classes.prediction import PredictionData as pred

import numpy as np

def parameters_piton_periodI():
    """Parameters pre-defined for unit tests."""
    # Excel file >> Piton de la Fournaise
    name_file = 'PitondelaFournaise_data'
    # PITON period 1: 1 to 74 | period 2: 74 to 120
    r1, rend = 1, 74

    qlong = 0.0024

    return name_file, [r1, rend], qlong

def test_get_data():
    """Test data collection function."""
    name_file, r, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.get_data(r[0], r[1])

    # size of the data
    assert len(edates) == 73
    assert len(evol) == 73
    assert len(cvol) == 74
    # check values
    assert evol[0] == 500000
    assert evol[1] == 12450000

    assert cvol[0] == 0
    assert cvol[1] == 500000
    assert cvol[-1] == 658060000


def test_prediction_data():
    """Test prediction data collection."""
    name_file, r, qlong = parameters_piton_periodI()

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.get_data(r[0], r[1])

    # last eruption ID (real data, prediction will be ID + 1)
    last_eruption = 5
    # start prediction instance (PredictionData)
    mypred = pred(edates, evol, cvol, idx=last_eruption)

    # check initial values
    assert mypred.in_evol == evol[:5]
    assert mypred.in_cvol == cvol[:6]
    assert mypred.n == len(edates[:5])

    # TODO move to UNIT test (sanity check)
    assert sum(mypred.dT_days) == mypred.dT_total
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




