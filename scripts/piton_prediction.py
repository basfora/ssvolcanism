
from classes.collectdata import VolcanoData as vd
from classes.basicfun import Basicfun as bf
from classes.prediction import PredictionData as pred
from classes.myplots import MyPlots

if __name__ == '__main__':

    # ------------------------------------------------------
    # Excel file >> Piton de la Fournaise PITON period 1: 1 to 74 | period 2: 74 to 120
    name_file = 'PitondelaFournaise_data'

    mp = MyPlots()
    # IMPORT REAL DATA
    piton = vd(name=name_file, printing=False)

    # get relevant data from the file
    edates, evol, cvol = piton.organize(period=1)

    start_after = 10
    stop_before = 12

    # ------------------------------------------------------

    # PREDICTION
    pp = pred(edates, evol, cvol, id_last=10)