from classes.collectdata import VolcanoData as vd
from classes.basicfun import Basicfun as bf
from classes.myplots import MyPlots



# TODO [1] for polyfit: MODIFY to force starting point to be cvol(T0)  | plot the same way | update numbers in writeup
# TODO [2] # can plot the Q line km3/year, for each year of the period | compute error, plot and update numbers (section 2.2)
# TODO [3] stochastic method

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
        piton.organize_period(period=p)

        # polyfit for period



        save_evol = f'{base_name}{p}_Evol'
        save_dT = f'{base_name}{p}_dT'
        # plot evolution volume and time interval
        mp.plot_set01(piton, plot_op=1, savename=save_evol)
        mp.plot_set01(piton, plot_op=2, savename=save_dT)