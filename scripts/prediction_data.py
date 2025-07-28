"""Prediction data generation and plot script"""
from classes.collectdata import VolcanoData as vd
from classes.myplots import MyPlots
from classes.prediction import PredictionData as pred

volcanoes = {'p': 'Piton', 'h': 'Hawaii', 'i': 'Iceland', 'g': 'Galapagos'}
volcano_id = volcanoes['h']
plot_show = False


if __name__ == '__main__':

    # Excel file
    name_file = f'Table{volcano_id}'

    # IMPORT REAL DATA
    vdata = vd(name=name_file, printing=False)
    eruptionsin = vdata.create_eruptions()

    start_after_eruption = 1  # first eruption ID
    stop_before_eruption = vdata.n  # last eruption ID
    last_id = start_after_eruption

    eruptionsout = [eruptionsin[0]]  # save first eruption
    # loop over eruptions
    while last_id < stop_before_eruption:

        # create subset of historical data until eruption # last_id
        vsub = vdata.create_subset(start_after_eruption, last_id)

        # create prediction instance for next eruption
        nextid = last_id + 1
        pp = pred(nextid)
        # save subset data
        pp.input_dataset(vsub)
        # input oe instance
        pp.input_eruption(eruptionsin[nextid - 1])

        # run methods to compute prediction
        oe = pp.run_methods()

        # save results
        eruptionsout.append(oe)

        # increment last_id
        last_id += 1


    # plot results
    mp = MyPlots(vdata.volcano_name)
    mp.volcano_nickname = volcano_id

    # QLINE PLOTS
    mp.linear_plots(eruptionsout, plot_show)

    # DETERMINISTIC PLOTS
    mp.sanity_check_det(eruptionsout)
    mp.det_plots(eruptionsout, plot_show)
    #
    # # STOCHASTIC PLOTS
    ids_to_plot = [i for i in range(2, vdata.n+1)]
    mp.sanity_check_stoc(eruptionsout)
    mp.stoc_plots(eruptionsout, ids_to_plot, plot_show)






