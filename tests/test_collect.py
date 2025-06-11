"""Test data collection functions."""
from classes.collectdata import VolcanoData as vd

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
