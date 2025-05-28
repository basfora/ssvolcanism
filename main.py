from collectdata import VolcanoData as vd
from computethings import ComputeThings as ct



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # import data from Excel file >> Piton de la Fournaise
    piton_data = vd(name='PitondelaFournaise_data', printing=True)
    # get data from the file
    piton_data.get_data()
    piton_data.set_long_term_rate(Q_long_term=0.0024)

    print('...\n...Getting simple stats...')

    idx0 = 0
    for i in range(2, piton_data.n+1):
        print('-----------------------------\nNew Eruption #', i)
        # simple stats for each eruption
        piton = ct(printing=True)
        idxf = i
        # export data for analysis as lists
        dates, eruptvol, cumvol = piton_data.output_rel_data(idx0, idxf)
        # set data for analysis
        piton.set_data(dates, eruptvol, cumvol)
        piton_data.add_rate_interval(piton.Qy, piton.avg_dt_days)

    print('==================================================')
    #print(piton_data.list_Q)
    piton_data.analyze_Q()
    print('==================================================')









