"""Place-holder for data specific scripts and plot generation"""

from classes.collectdata import VolcanoData as vd
from classes.eruption import OneEruption
from classes.prediction import PredictionData as pred
from classes.myplots import MyPlots as mp

# compute stats for Piton de la Fournaise volcano

if __name__ == '__main__':

    # ------------------------------------------------------
    # Excel file >> Piton de la Fournaise
    name_file = 'PitondelaFournaise_data'

    # IMPORT REAL DATA
    piton_data = vd(name=name_file, printing=False)
    # get relevant data from the file
    edates, evol, cvol = piton_data.organize(period=1)
    q_period = piton_data.Q1

    # save first eruption
    oe = OneEruption(eruption_id=1)
    oe.date.real, oe.evol.real, oe.cvol.real = edates[0], evol[0], cvol[0]
    eruptions = [oe]

    # LOOP OVER ERUPTIONS
    # last eruption ID (real data, prediction will be ID + 1)
    start_after_eruption = 1
    stop_before_eruption = 73


    last_eruption = start_after_eruption
    while last_eruption < stop_before_eruption:
        # ------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------
        pp = pred(edates, evol, cvol, id_last=last_eruption)
        pp.set_qperiod(q_period)

        # run prediction methods
        next_eruption = pp.run_methods()

        # ------------------------ quick analysis
        eruptions.append(next_eruption)
        # iterate to next eruption
        last_eruption += 1


    # PLOT
    my_plots = mp(piton=True)
    base_name = 'Piton_Period1_Cvol'
    my_plots.plot_set02(eruptions, base_name)
























