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
    eruptions = []

    # LOOP OVER ERUPTIONS
    # last eruption ID (real data, prediction will be ID + 1)
    start_after_eruption = 1
    stop_before_eruption = 119

    last_eruption = start_after_eruption
    while last_eruption < stop_before_eruption:

        # set q period (todo improve this, its messy)
        if last_eruption < 73:
            p = 1
            q_period = piton_data.Q1
        elif last_eruption == 73:
            p = 2
            q_period = piton_data.Q2

            last_eruption = 1
            stop_before_eruption = 46
        else:
            p = 2
            q_period = piton_data.Q2

        # get relevant data from the file
        edates, evol, cvol = piton_data.organize(p)  # period 0 for all data

        if last_eruption == 1:
            # save first eruption
            oe = OneEruption(eruption_id=1)
            oe.save_raw(edates[0], evol[0], cvol[1])
            eruptions.append(oe)

        # ------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------
        pp = pred(edates, evol, cvol, q_period, id_last=last_eruption)

        # run prediction methods
        next_eruption = pp.run_methods()

        # linear fit
        y_pt_line = piton_data.get_line_pt(pp.oe.id, 'linear')
        pp.oe.save_result(y_pt_line[1], pp.oe.dT.real, method=1)  # save linear extrapolation
        pp.oe.q_linear = piton_data.get_a_b(1)


        # save and iterate to next eruption
        eruptions.append(next_eruption)
        last_eruption += 1

    # ========================================================
    # PLOT
    # ========================================================
    if plot_flag:
        my_plots = mp(piton=True)
        # print for sanity check
        mp.sanity_check_det(eruptions)

        show_plot = False

        # DETERMINISTIC PLOTS
        mp.det_plots(eruptions, show_plot)

        # LINEAR REGRESSION PLOTS
        mp.linear_plots(eruptions, show_plot)

        # QLINE PLOTS
        base_name = 'Piton_Period0_Cvol_QLineError'
        mp.plot_volume_error(eruptions, 'cvol', 'qline', base_name, show_plot)























