from classes.collectdata import VolcanoData as vd
from classes.basicfun import Basicfun as bf


if __name__ == '__main__':

    # ------------------------------------------------------
    # Excel file >> Piton de la Fournaise PITON period 1: 1 to 74 | period 2: 74 to 120
    name_file = 'PitondelaFournaise_data'

    # real code starts here
    # ------------------------------------------------------
    # IMPORT REAL DATA Period I - (1936-1998)
    # init data collection instance (VolcanoData)
    piton_p1 = vd(name=name_file, printing=False)
    # get data from the file
    edates, evol, cvol = piton_p1.organize(period=1)