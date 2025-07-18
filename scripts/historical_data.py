"""Plot everything related to Piton volcano"""
from classes.collectdata import VolcanoData as vd
from classes.myplots import MyPlots

# ------------------------------------------------
# start with HISTORICAL DATA
# -------------------------------------------------

volcano_name = 'Piton'
plot_show = True

if __name__ == '__main__':

    # Excel file
    name_file = f'Table{volcano_name}'

    # IMPORT REAL DATA
    vdata = vd(name=name_file, printing=False)

    # to save plot
    base_name = volcano_name + '_Period'

    # todo change range to grab from vd.n_periods

    for period in range(3):
        vdata.organize(period)

        # plot evolution volume and time interval
        save_evol = f'{base_name}{period}_Evol'
        save_dT = f'{base_name}{period}_dT'

        mp = MyPlots()
        mp.plot_set01(vdata, 1, save_evol, plot_show)
        mp.plot_set01(vdata, 2, save_dT, plot_show)





