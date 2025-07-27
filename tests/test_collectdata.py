"""Test functions for VolcanoData class"""
import numpy as np
import datetime

from classes.collectdata import VolcanoData as vd
# from classes.subset import MySubset as ms

th1 = 1e-4
th2 = 1

# TRUE DATA
def true_data_piton():
    # true parameters
    vparam = dict()
    vparam['name'] = 'Piton de la Fournaise'
    vparam['n'] = 119
    vparam['n_periods'] = 2

    # first five rows of data
    first_five = dict()
    # eruption dates (edates) as datetime.date objects
    first_five['edates'] = [
        datetime.date(1936, 8, 1), datetime.date(1937, 8, 13),
        datetime.date(1938, 1, 1), datetime.date(1938, 12, 16),
        datetime.date(1942, 10, 1)]

    # eruption volumes (evol) in m3
    first_five['evols'] = [500000, 12450000, 4200000, 37350000, 1100000]

    # cumulative volumes (cvol) in m3
    first_five['cvols'] = [0, 500000, 12950000, 17150000, 54500000, 55600000]

    return vparam, first_five

def true_period1_piton():
    trueperiod = dict()
    trueperiod['label'] = 1
    trueperiod['volcano_name'] = 'Piton de la Fournaise'
    trueperiod['date_t0'] = datetime.date(1936, 8, 1)
    trueperiod['date_tf'] = datetime.date(1998, 3, 11)

    trueperiod['e0'] = 1
    trueperiod['ef'] = 73
    # number of eruptions
    trueperiod['n'] = 73

    # first/last two dates
    trueperiod['edates_first'] = [datetime.date(1936, 8, 1), datetime.date(1937, 8, 13)]
    trueperiod['edates_last'] = [datetime.date(1998, 3, 9), datetime.date(1998, 3, 11)]
    # first/last two volumes (m3)
    trueperiod['evols_first'] = [500000, 12450000]
    trueperiod['evols_last'] = [60000000, 2400000]
    # first/last two cumulative volumes (m3)
    trueperiod['cvols_first'] = [0, 500000]
    trueperiod['cvols_last'] = [655660000, 658060000]

    # cumulative volume at t0/tf
    trueperiod['cvol_t0'] = 0.0
    trueperiod['cvol_tf'] = 658060000
    trueperiod['cvol_delta'] = 658060000.0
    # rate for the period
    trueperiod['qyr'] = 0.0107
    trueperiod['q'] = 29295.003422313483

    # eruption volume stats
    trueperiod['evol_sum'] = 658060000
    trueperiod['evol_mean'] = 9014520.548
    trueperiod['evol_std'] = 12835882.39370088
    trueperiod['evol_median'] = 5400000
    trueperiod['evol_mode'] = 1933500

    # time stats
    trueperiod['dT_total_days'] = 22502
    trueperiod['dT_mean'] = 312
    trueperiod['dT_std'] = 424
    trueperiod['dT_median'] = 132
    trueperiod['dT_mode'] = 59
    # intervals between eruptions (first and last two)
    trueperiod['intervals_first'] = [377, 141]
    trueperiod['intervals_last'] = [2020, 2]
    # timeline of intervals (days)
    trueperiod['timeline_first'] = [0, 377, 518]
    trueperiod['timeline_last'] = [20480, 22500, 22502]

    return trueperiod

def true_period2_piton():
    trueperiod = dict()
    trueperiod['label'] = 2
    trueperiod['volcano_name'] = 'Piton de la Fournaise'
    trueperiod['date_t0'] = datetime.date(1999, 7, 19)
    trueperiod['date_tf'] = datetime.date(2018, 7, 13)

    trueperiod['e0'] = 74
    trueperiod['ef'] = 119
    # number of eruptions
    trueperiod['n'] = 46

    # first/last two dates
    trueperiod['edates_first'] = [datetime.date(1999, 7, 19), datetime.date(1999, 9, 28)]
    trueperiod['edates_last'] = [datetime.date(2018, 4, 27), datetime.date(2018, 7, 13)]
    # first/last two volumes (m3)
    trueperiod['evols_first'] = [1300000, 1400000]
    trueperiod['evols_last'] = [8000000, 300000]
    # first/last two cumulative volumes (m3)
    trueperiod['cvols_first'] = [658060000, 659360000]
    trueperiod['cvols_last'] = [1091410000, 1091710000]

    # cumulative volume at t0/tf
    trueperiod['cvol_t0'] = 658060000
    trueperiod['cvol_tf'] = 1091710000
    trueperiod['cvol_delta'] = 433650000
    # rate for the period
    trueperiod['qyr'] = 0.0228
    trueperiod['q'] = 62422.99794661191

    # eruption volume stats
    trueperiod['evol_sum'] = 433650000
    trueperiod['evol_mean'] = 9427173.913043479
    trueperiod['evol_std'] = 24370632.599864542
    trueperiod['evol_median'] = 2000000
    trueperiod['evol_mode'] = 4222500

    # time stats
    trueperiod['dT_total_days'] = 6934
    trueperiod['dT_mean'] = 154
    trueperiod['dT_std'] = 201
    trueperiod['dT_median'] = 103
    trueperiod['dT_mode'] = 35

    # intervals between eruptions (first and last two)
    trueperiod['intervals_first'] = [71, 138, 131]
    trueperiod['intervals_last'] = [264, 24, 77]
    # timeline of intervals (days)
    trueperiod['timeline_first'] = [0, 71, 209]
    trueperiod['timeline_last'] = [6833, 6857, 6934]

    return trueperiod

def true_period0_piton():
    """True parameters for period 0 (all data)"""
    # --------------------------------------------------------- period 0 (all data)
    trueperiod = dict()
    trueperiod['label'] = 0
    trueperiod['volcano_name'] = 'Piton de la Fournaise'
    trueperiod['date_t0'] = datetime.date(1936, 8, 1)
    trueperiod['date_tf'] = datetime.date(2018, 7, 13)

    trueperiod['e0'] = 1
    trueperiod['ef'] = 119
    # number of eruptions
    trueperiod['n'] = 119

    # first/last two dates
    trueperiod['edates_first'] = [datetime.date(1936, 8, 1), datetime.date(1937, 8, 13)]
    trueperiod['edates_last'] = [datetime.date(2018, 4, 27), datetime.date(2018, 7, 13)]
    # first/last two volumes (m3)
    trueperiod['evols_first'] = [500000, 12450000]
    trueperiod['evols_last'] = [8000000, 300000]
    # first/last two cumulative volumes (m3)
    trueperiod['cvols_first'] = [0, 500000]
    trueperiod['cvols_last'] = [1091410000, 1091710000]

    # cumulative volume at t0/tf
    trueperiod['cvol_t0'] = 0
    trueperiod['cvol_tf'] = 1091710000
    trueperiod['cvol_delta'] = 1091710000
    # rate for the period
    trueperiod['qyr'] = 0.013322210333767664
    trueperiod['q'] = 36474.224048645214

    # eruption volume stats
    trueperiod['evol_sum'] = 1091710000
    trueperiod['evol_mean'] = 9174033.613445379
    trueperiod['evol_std'] = 18185078.346992765
    trueperiod['evol_median'] = 4200000.0
    trueperiod['evol_mode'] = 4183500.0

    # time stats
    trueperiod['dT_total_days'] = 29931
    trueperiod['dT_mean'] = 253
    trueperiod['dT_std'] = 363
    trueperiod['dT_median'] = 119
    trueperiod['dT_mode'] = 59

    # intervals between eruptions (first, between periods and last two)
    trueperiod['intervals_first'] = [377, 141]
    trueperiod['intervals_middle'] = [2020, 2, 495, 71, 138]  # intervals between eruptions # 71, 72, 73, 74, 75, 76
    trueperiod['intervals_last'] = [24, 77]
    # timeline of intervals (days)
    trueperiod['timeline_first'] = [0, 377]
    trueperiod['timeline_middle'] = [20480, 22500, 22502, 22997, 23068]  # eruptions # 71, 72, 73, 74, 75
    trueperiod['timeline_last'] = [29854, 29931]

    return trueperiod

# ACTUAL TESTS
def test_import_to_lists():
    """Test with piton true parameters"""

    # import data from Piton de la Fournaise
    name_file = f'TablePiton'
    # import data
    vdata = vd(name=name_file, printing=False)
    # -------------------------------------------

    # true parameters and data
    vparam, first_five = true_data_piton()
    # check name
    assert vdata.file_name == name_file
    # check volcano name
    assert vdata.volcano_name == vparam['name']
    # check number of eruptions
    assert vdata.n == vparam['n']

    # DATA check first five rows of data
    assert vdata.edates_list[0] == first_five['edates'][0]
    assert vdata.edates_list[1] == first_five['edates'][1]
    assert vdata.edates_list[2] == first_five['edates'][2]
    assert vdata.edates_list[3] == first_five['edates'][3]
    assert vdata.edates_list[4] == first_five['edates'][4]

    assert vdata.evols_list[0] == first_five['evols'][0]
    assert vdata.evols_list[1] == first_five['evols'][1]
    assert vdata.evols_list[2] == first_five['evols'][2]
    assert vdata.evols_list[3] == first_five['evols'][3]
    assert vdata.evols_list[4] == first_five['evols'][4]

    assert vdata.cvols_list[0] == first_five['cvols'][0]
    assert vdata.cvols_list[1] == first_five['cvols'][1]
    assert vdata.cvols_list[2] == first_five['cvols'][2]
    assert vdata.cvols_list[3] == first_five['cvols'][3]
    assert vdata.cvols_list[4] == first_five['cvols'][4]
    assert vdata.cvols_list[5] == first_five['cvols'][5]

    return


def test_select_data():
    """Test selecting data from VolcanoData"""

    # import data from Piton de la Fournaise
    name_file = f'TablePiton'
    # import data
    vdata = vd(name=name_file, printing=False)

    # -------------------------------------------
    # true parameters and data
    vparam, first_five = true_data_piton()

    # -------------------------------------------
    # select data from eruption id0 (only)
    id0, idf = 5, None
    edate, evol, cvol = vdata.select_data(id0, idf)
    # change test if output changes to integer for one instance
    assert isinstance(edate, datetime.date)
    assert isinstance(evol, int)
    assert isinstance(cvol, int)
    # check actual data
    assert edate == first_five['edates'][-1] == datetime.date(1942, 10, 1)
    assert evol == first_five['evols'][-1] == 1100000
    assert cvol == first_five['cvols'][-1] == 55600000  # cvol at the end of eruption
    assert edate == vdata.edates_list[5 - 1]  # 5th eruption date
    assert evol == vdata.evols_list[5 - 1]  # 5th eruption date
    assert cvol == vdata.cvols_list[5]  # 5th eruption cumulative volume

    # -------------------------------------------
    # select data from eruptions id0 to idf (inclusive)
    id0, idf = 1, 5
    edates, evols, cvols = vdata.select_data(id0, idf)

    assert len(edates) == len(evols) == 5
    assert len(cvols) == 6  # cvol has one more element than evol
    # check actual data
    # edates
    assert edates[0] == first_five['edates'][0]
    assert edates[-1] == first_five['edates'][-1]
    # evol
    assert evols[0] == first_five['evols'][0]
    assert evols[-1] == first_five['evols'][-1]
    # cvol
    assert cvols[0] == first_five['cvols'][0]
    assert cvols[-1] == first_five['cvols'][-1]

    # -------------------------------------------
    # both None or -1, select data from start to end
    id0, idf = None, None
    edates_a, evol_a, cvol_a = vdata.select_data(id0, idf)
    edates_b, evol_b, cvol_b = vdata.select_data(-1, -1)
    # check that both methods return the same data
    assert len(edates_a) == len(edates_b) == vdata.n
    assert len(evol_a) == len(evol_b) == vdata.n
    assert len(cvol_a) == len(cvol_b) == vdata.n + 1  # cvol has one more element than evol
    # check actual data
    assert edates_a[0] == vdata.edates_list[0]
    assert edates_a[-1] == vdata.edates_list[-1]
    assert evol_a[0] == vdata.evols_list[0]
    assert evol_a[-1] == vdata.evols_list[-1]
    assert cvol_a[0] == vdata.cvols_list[0]
    assert cvol_a[-1] == vdata.cvols_list[-1]
    # check that both methods return the same data
    assert np.array_equal(edates_a, edates_b)
    assert np.array_equal(evol_a, evol_b)
    assert np.array_equal(cvol_a, cvol_b)

    # -------------------------------------------
    # id0=-1, select data from start to eruption idf (inclusive)
    id0, idf = -1, 5
    edates, evols, cvols = vdata.select_data(id0, idf)
    assert len(edates) == len(evols) == 5
    assert len(cvols) == 6  # cvol has one more element than evol
    # check actual data
    # edates
    assert edates[0] == first_five['edates'][0]
    assert edates[-1] == first_five['edates'][-1]
    # evol
    assert evols[0] == first_five['evols'][0]
    assert evols[-1] == first_five['evols'][-1]
    # cvol
    assert cvols[0] == first_five['cvols'][0] == 0.0    # cvol before eruption 1
    assert cvols[-1] == first_five['cvols'][-1]         # cvol at the end of eruption 5

    # -------------------------------------------
    # select data from id0 to end (inclusive)
    id0, idf = 5, -1
    edates, evols, cvols = vdata.select_data(id0, idf)
    assert len(edates) == len(evols) == vdata.n - 4  # 5th eruption to end
    assert len(cvols) == vdata.n - 4 + 1  # cvol has one more element than evol
    # check actual data
    assert np.array_equal(edates, vdata.edates_list[4:])  # from 5th eruption to end
    assert np.array_equal(evols, vdata.evols_list[4:])  # from 5th eruption to end
    assert np.array_equal(cvols, vdata.cvols_list[4:])  # from 5th eruption to end
    assert cvols[0] == 54500000  # cvol at the start of eruption 5
    assert cvols[1] == 55600000  # cvol at the end of eruption 5

    return


def test_periods():
    # import data from Piton de la Fournaise
    name_file = f'TablePiton'
    # import data
    vdata = vd(name=name_file, printing=False)
    # -------------------------------------------
    # true parameters and data
    vparam, first_five = true_data_piton()
    # -------------------------------------------
    # check number of periods
    assert vdata.n_periods == vparam['n_periods']

    for i in [0, 1, 2]:

        p = vdata.periods[i]
        if i ==0:
            myperiod = true_period0_piton()
        elif i == 1:
            myperiod = true_period1_piton()
        else:
            myperiod = true_period2_piton()

        assert p.label == i
        assert p.volcano_name == vparam['name']
        assert p.date_t0 == myperiod['date_t0']
        assert p.date_tf == myperiod['date_tf']
        assert p.e0 == myperiod['e0']
        assert p.ef == myperiod['ef']

        # first/last two dates
        assert p.edates[0] == myperiod['edates_first'][0]
        assert p.edates[1] == myperiod['edates_first'][1]
        assert p.edates[-2] == myperiod['edates_last'][-2]
        assert p.edates[-1] == myperiod['edates_last'][-1]

        # first/last two volumes (m3)
        assert p.evols[0] == myperiod['evols_first'][0]
        assert p.evols[1] == myperiod['evols_first'][1]
        assert p.evols[-2] == myperiod['evols_last'][-2]
        assert p.evols[-1] == myperiod['evols_last'][-1]

        # first/last two cumulative volumes (m3)
        assert p.cvols[0] == myperiod['cvols_first'][0]
        assert p.cvols[1] == myperiod['cvols_first'][1]
        assert p.cvols[-2] == myperiod['cvols_last'][-2]
        assert p.cvols[-1] == myperiod['cvols_last'][-1]

        # number of eruptions
        assert p.n == myperiod['n']

        # cumulative volume at t0/tf
        assert p.cvol_t0 == myperiod['cvol_t0']
        assert p.cvol_tf == myperiod['cvol_tf']
        assert p.cvol_delta == myperiod['cvol_delta']

        # rate for the period
        assert np.isclose(p.qyr, myperiod['qyr'], atol=th1)
        assert np.isclose(p.q, myperiod['q'], atol=th2)
        # eruption volume stats
        assert np.isclose(p.evol_sum, myperiod['evol_sum'], atol=th2)
        assert np.isclose(p.evol_mean, myperiod['evol_mean'], atol=th2)
        assert np.isclose(p.evol_std, myperiod['evol_std'], atol=th2)
        assert np.isclose(p.evol_median, myperiod['evol_median'], atol=th2)
        assert np.isclose(p.evol_mode, myperiod['evol_mode'], atol=th2)

        # time stats
        assert p.dT_total_days == myperiod['dT_total_days']
        assert np.isclose(p.dT_mean, myperiod['dT_mean'], atol=th2)
        assert np.isclose(p.dT_std, myperiod['dT_std'], atol=th2)
        assert np.isclose(p.dT_median, myperiod['dT_median'], atol=th2)
        assert np.isclose(p.dT_mode, myperiod['dT_mode'], atol=th2)
        # intervals between eruptions
        assert p.intervals[0] == myperiod['intervals_first'][0]
        assert p.intervals[1] == myperiod['intervals_first'][1]
        assert p.intervals[-2] == myperiod['intervals_last'][-2]
        assert p.intervals[-1] == myperiod['intervals_last'][-1]
        # timeline of intervals (days)
        assert p.timeline[0] == myperiod['timeline_first'][0]
        assert p.timeline[1] == myperiod['timeline_first'][1]
        assert p.timeline[-2] == myperiod['timeline_last'][-2]
        assert p.timeline[-1] == myperiod['timeline_last'][-1]

        # extra checks for period 0
        if i == 0:
            # middle (between periods) intervals
            # day of eruption # 71, 72, 73, 74, 75 (idx = ID - 1)
            assert p.timeline[70] == myperiod['timeline_middle'][0]
            assert p.timeline[71] == myperiod['timeline_middle'][1]
            assert p.timeline[72] == myperiod['timeline_middle'][2]
            assert p.timeline[73] == myperiod['timeline_middle'][3]
            assert p.timeline[74] == myperiod['timeline_middle'][4]
            # interval between eruptions # 71 and 72 (idx = IDlast - 2)
            assert p.intervals[70] == myperiod['intervals_middle'][0]
            # 72 and 73
            assert p.intervals[71] == myperiod['intervals_middle'][1]
            # 73 and 74
            assert p.intervals[72] == myperiod['intervals_middle'][2]
            # 74 and 75
            assert p.intervals[73] == myperiod['intervals_middle'][3]
            # 75 and 76
            assert p.intervals[74] == myperiod['intervals_middle'][4]





