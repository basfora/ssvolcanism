"""Plot Hisrtorical Data of each volcano"""
from classes.collectdata import VolcanoData as vd
from classes.myplots import MyPlots

volcanoes = {'p': 'Piton', 'h': 'Hawaii', 'i': 'Iceland', 'g': 'Galapagos'}
volcano_id = volcanoes['h']
plot_show = False

# ------------------------------------------------
# start with HISTORICAL DATA
# -------------------------------------------------
if __name__ == '__main__':

    # Excel file
    name_file = f'Table{volcano_id}'

    # IMPORT REAL DATA
    vdata = vd(name=name_file, printing=False)

    # to save plot
    base_name = volcano_id + '_Period'

    print(f"...Plotting historical data")
    for iperiod in vdata.periods:

        # get period data as subset
        mydata = vdata.periods[iperiod]

        # plot evolution volume and time interval
        save_evol = f'{base_name}{iperiod}_Evol'
        save_dT = f'{base_name}{iperiod}_dT'

        mp = MyPlots(vdata.volcano_name)
        mp.plot_set01(mydata, 1, save_evol, plot_show)
        mp.plot_set01(mydata, 2, save_dT, plot_show)





