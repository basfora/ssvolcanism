"""Place-holder for data specific scripts and plot generation"""

from classes.collectdata import VolcanoData as vd
from classes.eruption import OneEruption
from classes.prediction import PredictionData as pred
from classes.myplots import MyPlots as mp

# compute stats for Piton de la Fournaise volcano

if __name__ == '__main__':

    plot_flag = True

    # ------------------------------------------------------
    # Excel file >> Piton de la Fournaise
    name_file = 'PitondelaFournaise_data'

    # IMPORT REAL DATA
    piton_data = vd(name=name_file, printing=False)
    # get relevant data from the file
    edates, evol, cvol = piton_data.organize(period=0)  # period 0 for all data

    # save first eruption
    oe = OneEruption(eruption_id=1)
    oe.save_raw(edates[0], evol[0], cvol[1])
    oe.qperiod = piton_data.Q1
    eruptions = [oe]

    # LOOP OVER ERUPTIONS
    # last eruption ID (real data, prediction will be ID + 1)
    start_after_eruption = 1
    stop_before_eruption = 119

    last_eruption = start_after_eruption
    last_id = 1
    while last_eruption < stop_before_eruption:
        # ------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------
        edates, evol, cvol = piton_data.output_real_data(0, last_eruption)
        pp = pred(edates, evol, cvol, last_id)

        # set q period
        if last_id < 73:
            q_period = piton_data.Q1
        else:
            q_period = piton_data.Q2
        pp. set_period_info(q_period)

        enext, evolnext, cvolnext = piton_data.output_next(last_eruption)
        pp.save_real_next(enext, evolnext, cvolnext)

        # run prediction methods
        next_eruption = pp.run_methods()


        # save and iterate to next eruption
        eruptions.append(next_eruption)
        last_eruption += 1
        last_id += 1

    # ========================================================
    # PLOT
    # ========================================================
    if plot_flag:
        my_plots = mp(piton=True)
        # print for sanity check
        my_plots.sanity_check_det(eruptions)
        my_plots.sanity_check_stoc(eruptions)

        show_plot = False

        # DETERMINISTIC PLOTS
        #my_plots.det_plots(eruptions, show_plot)

























