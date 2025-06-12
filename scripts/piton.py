"""Place-holder for data specific scripts and plot generation"""

from classes.collectdata import VolcanoData as vd
from classes.basicfun import Basicfun as bf
from classes.prediction import PredictionData as pred

import numpy as np

# compute stats for Piton de la Fournaise volcano, period 1 (1936-1998)

if __name__ == '__main__':

    # ------------------------------------------------------
    # Excel file >> Piton de la Fournaise (todo: change this to be a parameter inside VolcanoData)
    name_file = 'PitondelaFournaise_data'
    # PITON period 1: 1 to 74 | period 2: 74 to 120
    r1, rend = 1, 74
    period = '1'
    # real code starts here
    # ------------------------------------------------------
    # IMPORT REAL DATA PER PERIOD
    # init data collection instance (VolcanoData)
    piton_data = vd(name=name_file, printing=False)
    piton_data.piton_rates()
    # get data from the file
    edates, evol, cvol = piton_data.organize(r1, rend)
    qtheory = piton_data.output_Q(period)

    # LOOP OVER ERUPTIONS
    # last eruption ID (real data, prediction will be ID + 1)
    start_after_eruption = 10
    stop_before_eruption = 11
    error_evol = []

    last_eruption = start_after_eruption
    while last_eruption < stop_before_eruption:
        # ------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------
        mp = pred(edates, evol, cvol, idx=last_eruption, qtheory=qtheory)
        mp.run_prediction_methods()
        mp.organize_stuff()

        # ------------------------ quick analysis
        error_evol.append(mp.error_evol_per)
        # iterate to next eruption
        last_eruption += 1

    # ---------------------------- quick analysis
    bf.print_mark()
    print(f"Error EVOL(t2) %: \nMEAN: {np.mean(error_evol):.1f} | MAX {max(error_evol):.1f}| MIN {min(error_evol):.1f} ")
    j = start_after_eruption
    for er in error_evol:
        j += 1
        print(f"({j}) {er:.2f} %", end=" | ")




















