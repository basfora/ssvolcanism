"""Place-holder for data specific scripts and plot generation"""

from classes.collectdata import VolcanoData as vd
from classes.computethings import ComputeThings as ct
from classes.basicfun import basicfun as bf
from classes.myplots import MyPlots as mp

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# compute stats for Piton de la Fournaise volcano, period 1 (1936-1998)

if __name__ == '__main__':

    name_file = 'PitondelaFournaise_data'

    # import data from Excel file >> Piton de la Fournaise
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    # period 1: 1 to 74 | period 2: 74 to 120
    r1, rend = 1, 120
    piton_data.get_data(r1, rend)

    # whole period data
    idxf = piton_data.n - 1
    edates, evol, cvol = piton_data.output_rel_data(0, piton_data.n)
    # print period and number of eruptions
    print('==================================================')

    # plot eruption volumes
    mp.plot_data(edates, evol)

    mp.plot_cvol(edates, evol, cvol)
















