"""Place-holder for data specific scripts and plot generation"""

from classes.collectdata import VolcanoData as vd
from classes.computethings import ComputeThings as ct
from classes.basicfun import basicfun as bf
from classes.myplots import MyPlots as mp
from classes.prediction import PredictionData as pred

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# compute stats for Piton de la Fournaise volcano, period 1 (1936-1998)

if __name__ == '__main__':

    # IMPORT REAL DATA (BY PERIOD OR ALL)
    # ------------------------------------------------------
    # Excel file >> Piton de la Fournaise
    name_file = 'PitondelaFournaise_data'
    # PITON period 1: 1 to 74 | period 2: 74 to 120
    r1, rend = 1, 74

    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_data.get_data(r1, rend)
    # ------------------------------------------------------

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------
    # last eruption ID (real data, prediction will be ID + 1)
    last_eruption = 5
    # start prediction instance (PredictionData)
    mypred = pred(edates, evol, cvol, idx=last_eruption)
    mypred.one_step_ahead()


















