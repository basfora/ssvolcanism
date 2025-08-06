"""Prediction data generation and plot script"""
from classes.collectdata import VolcanoData as vd
from classes.myplots import MyPlots

volcanoes = {'p': 'Piton', 'h': 'Hawaii', 'i': 'Iceland', 'g': 'Galapagos'}
volcano_id = volcanoes['g']
plot_show = False
print_oe = False


if __name__ == '__main__':

    # Excel file
    name_file = f'Table{volcano_id}'

    # IMPORT REAL DATA
    vdata = vd(name=name_file, printing=False)
    eruptionsin = vdata.create_eruptions_hash(print_oe)

    start_after_eruption = 1  # first eruption ID
    stop_before_eruption = vdata.n  # last eruption ID

    # loop over eruptions
    last_id = start_after_eruption
    # save first eruption
    eruptionsout = [eruptionsin[1]]
    while last_id < stop_before_eruption:

        # create subset of historical data until eruption # last_id
        vsub = vdata.create_subset(start_after_eruption, last_id)

        # create prediction instance for next eruption
        nextid = last_id + 1
        nextoe = eruptionsin[nextid]  # next eruption instance

        # run prediction methods
        oe = vsub.run_prediction(nextoe)

        # save modified instance
        eruptionsout.append(oe)

        # increment last_id
        last_id += 1

    # -------------------------------------------------
    # PLOT RESULTS
    # -------------------------------------------------
    mp = MyPlots(vdata.volcano_name)
    mp.volcano_nickname = volcano_id

    # QLINE PLOTS
    mp.sanity_check(eruptionsout, 'qline')
    mp.linear_plots(eruptionsout, plot_show)


    # DETERMINISTIC PLOTS
    mp.sanity_check(eruptionsout, 'det')
    mp.det_plots(eruptionsout, plot_show)
    #
    # # STOCHASTIC PLOTS
    ids_to_plot = [i for i in range(2, vdata.n+1)]
    mp.sanity_check_stoc(eruptionsout)
    mp.stoc_plots(eruptionsout, ids_to_plot, plot_show)






