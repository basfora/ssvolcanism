from classes.collectdata import VolcanoData as vd
from classes.basicfun import Basicfun as bf
from classes.myplots import MyPlots

if __name__ == '__main__':

    # ------------------------------------------------------
    # Excel file >> Piton de la Fournaise PITON period 1: 1 to 74 | period 2: 74 to 120
    name_file = 'PitondelaFournaise_data'

    mp = MyPlots()
    # IMPORT REAL DATA
    piton = vd(name=name_file, printing=False)
    # TO SAVE PLOT
    base_name = 'Piton_Period'

    # Period 1
    for p in range(3):
        piton.organize(period=p)
        save_evol = f'{base_name}{p}_Evol'
        save_dT = f'{base_name}{p}_dT'
        # plot evolution volume (evol)
        mp.plot_set01(piton, plot_op=1, savename=save_evol)
        mp.plot_set01(piton, plot_op=2, savename=save_dT)

