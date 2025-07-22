"""Plot everything related to Piton volcano"""
from classes.collectdata import VolcanoData as vd
from classes.myplots import MyPlots

# ------------------------------------------------
# start with HISTORICAL DATA
# -------------------------------------------------
volcanoes = {'p': 'Piton', 'h': 'Hawaii', 'i': 'Iceland', 'g':'Galapagos'}
volcano_name = volcanoes['p']
plot_show = False

if __name__ == '__main__':

    # Excel file
    name_file = f'Table{volcano_name}'

    # IMPORT REAL DATA
    vdata = vd(name=name_file, printing=False)

    # to save plot
    base_name = volcano_name + '_Period'

    for period in range(vdata.n_periods + 1):
        vdata.organize_period(period)

        # plot evolution volume and time interval
        save_evol = f'{base_name}{period}_Evol'
        save_dT = f'{base_name}{period}_dT'

        mp = MyPlots(name_file)
        mp.plot_set01(vdata, 1, save_evol, plot_show)
        mp.plot_set01(vdata, 2, save_dT, plot_show)





