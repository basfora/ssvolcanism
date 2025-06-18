"""Place-holder for data specific scripts and plot generation"""

from classes.collectdata import VolcanoData as vd
from classes.basicfun import Basicfun as bf
from classes.prediction import PredictionData as pred

import numpy as np

# compute stats for Piton de la Fournaise volcano

if __name__ == '__main__':

    # ------------------------------------------------------
    # Excel file >> Piton de la Fournaise
    name_file = 'PitondelaFournaise_data'

    # IMPORT REAL DATA
    piton_data = vd(name=name_file, printing=False)
    # get relevant data from the file
    edates, evol, cvol = piton_data.organize(period=1)
    q_period = piton_data.output_Q()


    # LOOP OVER ERUPTIONS
    # last eruption ID (real data, prediction will be ID + 1)
    start_after_eruption = 1
    stop_before_eruption = 73
    error_evol = []

    last_eruption = start_after_eruption
    while last_eruption < stop_before_eruption:
        # ------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------
        pp = pred(edates, evol, cvol, id_last=last_eruption)
        pp.set_qperiod(q_period)

        # run prediction methods
        pp.run_methods()

        # ------------------------ quick analysis
        error_evol.append(pp.oe.evol.deterministic_error.per)
        # iterate to next eruption
        last_eruption += 1

    # ---------------------------- quick analysis
    bf.print_mark()
    print(f"Error EVOL(t2) %: \nMEAN: {np.mean(error_evol):.1f} | MAX {max(error_evol):.1f}| MIN {min(error_evol):.1f} ")
    j = start_after_eruption
    for er in error_evol:
        j += 1
        print(f"({j}) {er:.2f} %", end=" | ")




















