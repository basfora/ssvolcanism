"""Importing all data from Excel files and saving them
Return 3 lists: edates, evol, cvol"""
from classes.basicfun import Basicfun as bf

import pandas as pd
import os


class VolcanoData:
    """Class to import data from excel files"""

    def __init__(self, name=None, printing=False):

        self.printing = printing
        self.save_plot = True
        # file name to be imported
        self.name = ''
        self.extension = '.xlsx'
        self.raw_data_dir = 'rawdata'
        self.plot_dir = 'plots'
        # --------------------------------
        # PATHS
        # --------------------------------
        # current directory
        self.current_dir = ''
        # path to the raw data folder (where Excel file is)
        self.raw_data_path = ''
        self.path_to_file = ''

        # --------------------------------
        # DATA HANDLING
        # --------------------------------
        # original data frames
        self.df_volcano = None
        self.series_date = None
        self.series_eruptvol = None
        self.series_cumvol = None

        # as lists (to be outputed)
        self.list_date = []
        self.list_eruptvol = []
        self.list_cumvol = []

        # --------------------------------
        # for computing stuff
        self.n = 0  # number of eruptions
        # rates
        self.Q_long = None
        self.Q1 = None  # rate for period 1
        self.Q2 = None  # rate for period 2
        self.Q3 = None  # rate for other period

        # --------------------------------
        # INIT FUNCTIONS
        # --------------------------------
        self.set_name(name)
        # ---------------------------------

    def set_name(self, name=None):
        """Set the name of the file to be imported"""
        if name is None:
            # if no name is given, ask for it
            name = input("Enter the name of the file: ")
        self.name = name

    # UT - OK
    def collect_from(self):
        """Return path to Excel file: current > parent > rawData"""

        # Get the current working directory
        self.current_dir = os.getcwd()
        # Get the parent directory > '../PycharmProjects/volcano'
        parent_dir = os.path.dirname(self.current_dir)
        # fix path for when running unit tests
        if '/volcano' not in parent_dir:
            parent_dir += '/volcano'
        # Get the path to the rawData folder > '../PycharmProjects/volcano/rawdata'
        self.raw_data_path = os.path.join(parent_dir, self.raw_data_dir)
        # path to file
        self.path_to_file = os.path.join(self.raw_data_path, self.name)
        # add extension (.xlsx) if not present
        if ".xlsx" not in self.path_to_file:
            self.path_to_file += self.extension

        return self.path_to_file

    # UT - OK
    def import_data(self):
        """Import data from an Excel file and return as DataFrame"""

        # get paths and extensions
        self.collect_from()
        # Read the Excel file
        df = pd.read_excel(self.path_to_file, date_format='%d/%m/%Y')

        return df

    # UT - OK
    def organize(self, r1=1, rend=74):
        """Get data from file and return it as a list
        default: PitondelaFournaise
        years: 1936 - March 1998
        :param r1: row to start from (default 1), period II: 74
        :param rend: row to end at (default 74), period II: 120"""
        # ----------------------------------
        # columns to use: B and E
        cDate, cErupted, cCV = 1, 3, 4
        # ----------------------------------
        # import dall ata from the file
        self.df_volcano = self.import_data()
        # -----------------------------------
        # separate relevant data from the file, as DataFrames
        self.series_date = self.df_volcano.iloc[r1:rend, cDate]
        self.series_eruptvol = self.df_volcano.iloc[r1:rend, cErupted]
        self.series_cumvol = self.df_volcano.iloc[r1:rend, cCV]
        self.n = len(self.series_eruptvol)  # number of eruptions
        # -----------------------------------
        # turn into lists to make it easier to see/plot
        self.list_eruptvol = self.series_eruptvol.tolist()

        # adjust for fake zero init
        self.list_cumvol = [0] + self.series_cumvol.tolist()
        self.list_date = self.series_date.tolist()

        # only need date and cumulative volume
        return self.list_date, self.list_eruptvol, self.list_cumvol,

    def output_rel_data(self, idx_0=None, idx_f=None):
        """Output relevant data for analysis as lists"""
        if idx_0 is None:
            idx_0 = 0
        if idx_f is None:
            idx_f = self.n - 1

        rel_dates = self.list_date[idx_0:idx_f]
        rel_cumvol = self.list_cumvol[idx_0:idx_f]
        rel_eruptvol = self.list_eruptvol[idx_0:idx_f]

        return rel_dates, rel_eruptvol, rel_cumvol

    def set_Q(self, q: float, which='long'):
        """Set the rate of eruptions
        :param q: rate of eruptions in km3/yr
        :param which: 'long' for long-term,
                    '1' for period 1, '2' for period 2"""
        if which == 'long':
            self.Q_long = q
        elif which == '1':
            self.Q1 = q
        elif which == '2':
            self.Q2 = q
        else:
            print(f"Unknown rate type: {which}. Use 'long', '1', or '2'.")

    def output_Q(self, which='long'):
        """Output the rate of eruptions"""
        if which == 'long':
            return self.Q_long
        elif which == '1':
            return self.Q1
        elif which == '2':
            return self.Q2
        else:
            print(f"Unknown rate type: {which}. Use 'long', '1', or '2'.")
            return None

    def piton_rates(self):
        """Set the rates of eruptions for PitondelaFournaise
        Source: manuscript by Galetto (2025), Q unit: km3/yr
        Q unit for computation: m3/day"""

        # Transform km3/yr ro m3/day
        qlong = bf.Qy_to_Qday(0.0024)
        qperiod1 = bf.Qy_to_Qday(0.0107)
        qperiod2 = bf.Qy_to_Qday(0.0228)

        # long-term rate
        self.set_Q(qlong, 'long')
        # period 1 rate
        self.set_Q(qperiod1, '1')
        # period 2 rate
        self.set_Q(qperiod2, '2')


if __name__ == "__main__":
    # create an instance of the class for a volcano
    piton = VolcanoData(name='PitondelaFournaise_data', printing=True)
    # get data from the file
    piton.organize()
