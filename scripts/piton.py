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
    oe.date.real, oe.evol.real, oe.cvol.real = edates[0], evol[0], cvol[1]
    eruptions = [oe]

    # LOOP OVER ERUPTIONS
    # last eruption ID (real data, prediction will be ID + 1)
    start_after_eruption = 1
    stop_before_eruption = 119

    last_eruption = start_after_eruption
    while last_eruption < stop_before_eruption:
        # ------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------
        pp = pred(edates, evol, cvol, id_last=last_eruption)
        if last_eruption < 73:
            q_period = piton_data.Q1
        else:
            q_period = piton_data.Q2

        pp.set_qperiod(q_period)

        # run prediction methods
        next_eruption = pp.run_methods()

        # linear fit
        y_pt_line = piton_data.get_line_pt(pp.oe.id, 1)
        pp.oe.save_result(y_pt_line[1], pp.oe.dT.real, method=1)  # save linear extrapolation
        pp.oe.q_linear = piton_data.get_a_b(1)

        y_pt_qline = piton_data.get_line_pt(pp.oe.id, 0)
        pp.oe.save_result(y_pt_line[1], pp.oe.dT.real, method=0)  # save q-linear extrapolation
        pp.oe.q_line_xy = piton_data.get_a_b(0)

        # ------------------------ quick analysis
        eruptions.append(next_eruption)
        # iterate to next eruption
        last_eruption += 1

    # ========================================================
    # PLOT
    # ========================================================
    if plot_flag:
        my_plots = mp(piton=True)
        # print for sanity check
        mp.sanity_check_det(eruptions)

        show_plot = False

        # Plot 1: CVOL real vs expected (DET)
        base_name = 'Piton_Period0_Cvol_Det'
        my_plots.plot_real_vs_expected(eruptions, 'cvol',
                                       base_name, show_plot)

        # Plot 2: CVOL error
        base_name = 'Piton_Period0_Cvol_DetError'
        my_plots.plot_volume_error(eruptions, 'cvol', 'det',
                                   base_name, show_plot)

        # Plot 3: EVOL real vs expected
        base_name = 'Piton_Period0_Evol_Det'
        my_plots.plot_real_vs_expected(eruptions, 'evol',
                                       base_name, show_plot)

        # Plot 4: EVOL error
        base_name = 'Piton_Period0_Evol_DetError'
        my_plots.plot_volume_error(eruptions, 'evol', 'det',
                                   base_name, show_plot)

        #show_plot = True
        # Plot 5: CVOL real vs expected (LINEAR)
        base_name = 'Piton_Period0_Cvol_Linear'
        my_plots.plot_linear(eruptions, 'cvol', 'linear',
                             base_name, show_plot)

        base_name = 'Piton_Period0_Cvol_LinearError'
        my_plots.plot_volume_error(eruptions, 'cvol', 'linear',
                                   base_name, show_plot)

    # mean + std == 64% samples (double check that in my thesis)
























