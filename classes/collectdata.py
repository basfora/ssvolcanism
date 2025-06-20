"""Importing all data from Excel files and saving them
Return 3 lists: edates, evol, cvol"""
import datetime

from classes.basicfun import Basicfun as bf
import numpy as np

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

        # Piton specific
        self.period = None # 0 for all data, 1 for period 1, 2 for period 2
        self.r1, self.rend = 0, 0 # row to start and end collection of data
        self.date0 = None # start of period

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
        self.Qlong = None
        self.Q1 = None  # rate for period 1
        self.Q2 = None  # rate for period 2
        self.Q3 = None  # rate for other period

        # --------------------------------
        # To Have for plotting (might delete later)
        self.mean_evol = None  # mean eruption volume
        self.std_evol = None
        self.median_evol = None  # median eruption volume
        self.mode_evol = None

        self.intervals = []  # list of intervals between eruptions
        self.timeline = []  # timeline of eruptions
        self.line_points = None  # points for linear extrapolation


        self.mean_dT = None  # mean eruption interval
        self.std_dT = None
        self.median_dT = None  # median eruption interval
        self.mode_dT = None


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
    def organize(self, period=1):
        """Get data from file and return it as a list
        default: PitondelaFournaise
        years: 1936 - March 1998
        :param period: 1 for period I (1936-1998), 2 for period II (1999-2023), 0 for all data
        """
        # ----------------------------------

        # :param r1: row to start from (default 1), period II: 74
        #         :param rend: row to end at (default 74), period II: 120

        # set rows to use according to the period
        if 'Piton' in self.name:
            self.piton_rates()
            # Define period (I, II or both)
            if period == 1:
                self.period = 1
                r1, rend = 1, 74
                self.date0 = datetime.date(1936,1,8)
                cvolT0 = 0
            elif period == 2:
                self.period = 2
                r1, rend = 74, 120
                self.date0 = datetime.date(1999, 7, 19)
                cvolT0 = 659360000
            else:
                self.period = 0
                r1, rend = 1, 120
                self.date0 = datetime.date(1936, 1, 8)
                cvolT0 = 0
        else:
            # todo under construction
            # for other volcanoes, set default values
            r1, rend = 1, 74
            cvolT0 = 0

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
        self.list_cumvol = [cvolT0] + self.series_cumvol.tolist()
        self.list_date = [d.date() for d in self.series_date]

        # only need date and cumulative volume
        return self.list_date, self.list_eruptvol, self.list_cumvol,

    def compute_for_plotting(self):
        """Compute mean, median and mode for plotting"""

        # mean, median and mode for eruption volumes
        self.mean_evol, self.std_evol = bf.compute_mean_std(self.list_eruptvol)
        self.median_evol = bf.compute_median(self.list_eruptvol)

        # intervals between eruptions
        self.intervals = bf.compute_intervals(self.list_date)
        self.timeline = bf.compute_timeline(self.intervals, 0)  # start from 1

        # mean, median and mode for eruption intervals
        self.mean_dT, self.std_dT = bf.compute_mean_std(self.intervals)
        self.median_dT = bf.compute_median(self.intervals)


    def linear_extrapolation(self):
        """Linear extrapolation of eruption volumes and cumulative volumes"""

        self.compute_for_plotting()
        # use timeline to fit to line
        xvalues = self.timeline
        yvalues = self.list_cumvol[1:]

        # force the first point to be the cumulative volume at T0
        # x0 = xvalues[0]
        # y0 = yvalues[0]
        #
        # xvalues_adj = [x - x0 for x in xvalues]  # adjust x values to start from 0
        # yvalues_adj = [y - y0 for y in yvalues]  # adjust x values to start from 0
        #
        # a, _ = np.polyfit(xvalues_adj, yvalues_adj, 1)
        # b = y0 - a * x0  # compute b to ensure the line passes through (x0, y0)


        # linear squares fit (no forced initial point) -- uncomment
        a, b = np.polyfit(xvalues, yvalues, 1)
        self.a, self.b = a, b

        # create points for the line based on fit and timeline
        self.line_points = [(x, a * x + b) for x in self.timeline]

    def get_line_pt(self, eruption_id: int):
        """Get the point on the line for a given eruption ID"""
        if not self.line_points:
            self.linear_extrapolation()

        return self.line_points[eruption_id - 1]

    def get_a_b(self):
        return self.a, self.b



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

    def set_Q(self, q: float, which: int):
        """Set the rate of eruptions
        :param q: rate of eruptions in km3/yr
        :param which: 'long' for long-term,
                    '1' for period 1, '2' for period 2"""
        if which == -1:
            self.Qlong = q
        elif which == 1:
            self.Q1 = q
        elif which == 2:
            self.Q2 = q
        else:
            print(f"Unknown rate type: {which}. Use 'long', '1', or '2'.")

    def output_Q(self, which=None):
        """Output the rate of eruptions"""

        if which is None:
            if self.period == 1:
                return self.Q1
            elif self.period == 2:
                return self.Q2
            elif self.period == 0:
                return 'will computer later (todo)'
            else:
                return None

        elif which == -1:
            return self.Qlong
        else:
            return None

    def piton_rates(self):
        """Set the rates of eruptions for PitondelaFournaise
        Source: manuscript by Galetto (2025), Q unit: km3/yr
        Q unit for computation: m3/day"""

        # Transform km3/yr ro m3/day
        self.Q1 = bf.Qy_to_Qday(0.0107)
        self.Q2 = bf.Qy_to_Qday(0.0228)
        self.Qlong = bf.Qy_to_Qday(0.0024)



if __name__ == "__main__":
    # create an instance of the class for a volcano
    piton = VolcanoData(name='PitondelaFournaise_data', printing=True)
    # get data from the file
    piton.organize()
