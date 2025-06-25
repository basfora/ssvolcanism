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

    # LOOP OVER ERUPTIONS - Period I
    # last eruption ID (real data, prediction will be ID + 1)
    start_after_eruption = 1
    stop_before_eruption = 73
    p, q_period = 1, piton_data.Q1

    # get relevant data from the file
    piton_data.organize(p)  # period 0 for all data
    # get qline - period I
    piton_data.linear_extrapolation(3)

    last_eruption = start_after_eruption
    last_id = 1
    while last_eruption < stop_before_eruption:

        # get relevant data from the file
        edates, evol, cvol = piton_data.output_real_data(0, last_eruption)

        if last_eruption == start_after_eruption:
            # save first eruption
            oe1 = OneEruption(last_eruption)
            oe1.save_raw(edates[0], evol[0], cvol[1])
            oe1.qperiod = piton_data.Q1

            # # save first eruption
            # pt_qline = piton_data.get_line_pt(0, 'qline')
            # oe1.save_result(pt_qline[1], oe1.dT.t2, method=4)

            eruptions.append(oe1)

        # ------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------
        pp = pred(edates, evol, cvol, q_period, last_id)
        pp.set_period_info(piton_data.Q1)

        #if last_eruption < stop_before_eruption:
        enext, evolnext, cvolnext = piton_data.output_next(last_eruption)
        pp.save_real_next(enext, evolnext, cvolnext)


        # run prediction methods
        next_eruption = pp.run_methods()

        # linear fit
        pt_qline = piton_data.get_line_pt(last_eruption, 'qline')
        pp.oe.save_result(pt_qline[1], pp.oe.dT.t2, method=4)  # save qline pt
        pp.oe.a, pp.oe.b = piton_data.get_a_b()

        # save and iterate to next eruption
        eruptions.append(next_eruption)
        last_eruption += 1
        last_id += 1

    #-------------------------------------------------------
    # period 2
    start_after_eruption = 1
    stop_before_eruption = 46
    p, q_period = 2, piton_data.Q2

    # get relevant data from the file
    piton_data.organize(p)  # period 0 for all data
    # get qline - period I
    piton_data.linear_extrapolation(3)

    last_eruption = start_after_eruption
    last_id = 73
    while last_eruption < stop_before_eruption:

        # get relevant data from the file
        edates, evol, cvol = piton_data.output_real_data(0, last_eruption)

        if last_id == 73:
            # save first eruption -- # 74
            oe1 = OneEruption(last_id + 1)
            oe1.save_raw(edates[0], evol[0], cvol[1])
            oe1.qperiod = piton_data.Q2  # set qperiod
            # # save first eruption
            # pt_qline = piton_data.get_line_pt(0, 'qline')
            # oe1.save_result(pt_qline[1], oe1.dT.t2, method=4)

            eruptions.append(oe1)
            last_id += 1

        # ------------------------------------------------------
        # PREDICTION
        # ------------------------------------------------------
        pp = pred(edates, evol, cvol, q_period, last_id)
        pp.set_period_info(piton_data.Q2)

        enext, evolnext, cvolnext = piton_data.output_next(last_eruption)
        pp.save_real_next(enext, evolnext, cvolnext)

        # run prediction methods
        next_eruption = pp.run_methods()

        # linear fit
        pt_qline = piton_data.get_line_pt(last_eruption, 'qline')
        pp.oe.save_result(pt_qline[1], pp.oe.dT.t2, method=4)  # save qline pt
        pp.oe.a, pp.oe.b = piton_data.get_a_b()

        # save and iterate to next eruption
        eruptions.append(next_eruption)
        last_eruption += 1
        last_id += 1


    # ========================================================
    # PLOT
    # ========================================================
    if plot_flag:
        mpp = mp(piton=True)
        # print for sanity check
        mpp.sanity_check_det(eruptions)

        show_plot = False

        # DETERMINISTIC PLOTS
        mpp.det_plots(eruptions, show_plot)

        # LINEAR REGRESSION PLOTS
        mpp.linear_plots(eruptions, show_plot)





















