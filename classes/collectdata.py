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

        #---------------------------------
        # source file is formatted
        #---------------------------------
        # TODO add here new info to expand to all volcanoes




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

        self.volcano_name = [
            'Piton de la Fournaise',
            'Hawaii',
            'Iceland',
            'Western Galapagos']

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
        self.import_data()
        # ---------------------------------

    def set_name(self, name=None):
        """Set the name of the file to be imported"""

        """Set the name of the period"""
        if 'piton' in name.lower():
            self.name = 'Piton de la Fournaise'
        elif 'hawaii' in name.lower():
            self.name = 'Hawaii'
        elif 'iceland' in name.lower():
            self.name = 'Iceland'
        elif 'galapagos' in name.lower():
            self.name = 'Western Galapagos'
        else:
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
        # Read the Excel file (all of it)
        self.df_volcano = pd.read_excel(self.path_to_file, date_format='%d/%m/%Y')

        self.extract_parameters()

        # todo remove this when all volcanoes are implemented
        if 'Piton' in self.name:
            self.piton_rates()

        return self.df_volcano

    def extract_parameters(self):
        """Extract parameters from the DataFrame for the volcano"""

        # columns of interest - Periods info
        cperiod = 6
        cID0, cIDf = 7, 8
        cedatet0, cedatetf = 9, 10
        cQ = 11  # rate in km3/yr

        # less to write
        mydf = self.df_volcano
        self.periods = []  # list to store periods
        row = 1
        while True:
            pi = mydf.iat[row, cperiod]  # period number
            if pd.isna(pi):
                break
            myperiod = OfficialPeriod(pi)
            # get paramters for the period
            datet0, datetf = mydf.iat[row, cedatet0], mydf.iat[row, cedatetf]
            eid_t0, eid_tf = mydf.iat[row, cID0], mydf.iat[row, cIDf]
            q = mydf.iat[row, cQ]

            # save them in the period instance
            myperiod.set_dates(datet0.date(), datetf.date())
            myperiod.set_eIDs(eid_t0, eid_tf)
            myperiod.set_q(q)

            # store and print
            self.periods.append(myperiod)
            print(f'Saved Period {myperiod.number}: {myperiod.date_t0} - {myperiod.date_tf} (eruptions {myperiod.e0} - {myperiod.ef}), Q = {q} km3/yr')
            row += 1


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
                cvolT0 = 658060000
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


    def linear_extrapolation(self, opt=1):
        """Linear extrapolation of eruption volumes and cumulative volumes"""

        self.compute_for_plotting()
        # use timeline to fit to line
        xvalues = self.timeline
        yvalues = self.list_cumvol[1:]


        if opt == 1:
            # no forced initial point, just fit the line
            a, b = np.polyfit(xvalues, yvalues, 1)

        elif opt == 2:
            # force the first point to be the cumulative volume at T0
            x0 = xvalues[0]
            y0 = yvalues[0]

            xvalues_adj = [x - x0 for x in xvalues]  # adjust x values to start from 0
            yvalues_adj = [y - y0 for y in yvalues]  # adjust x values to start from 0

            a, _ = np.polyfit(xvalues_adj, yvalues_adj, 1)
            b = y0 - a * x0  # compute b to ensure the line passes through (x0, y0)
        else:
            x1, y1 = xvalues[0], yvalues[0]
            x2, y2 = xvalues[-1], yvalues[-1]
            a = (y2 - y1) / (x2 - x1)  # slope
            b = y1 - a * x1  # intercept

        # save the slope and intercept
        self.a, self.b = a, b

        # create points for the line based on fit and timeline
        self.line_points = [(x, a * x + b) for x in self.timeline]

    def get_line_pt(self, eruption_idx: int, method='qline'):
        """Get the point on the line for a given eruption ID"""

        if method == 'linear':
            if not self.line_points:
                self.linear_extrapolation(1)
            return self.line_points[eruption_idx]

        elif method == 'qline':
            if not self.line_points:
                self.linear_extrapolation(3)
            return self.line_points[eruption_idx]


        else:
            print(f"Unknown method {method}. Use 1 for linear extrapolation or 2 for q line.")
            return None



    def get_a_b(self):
        return self.a, self.b



    def output_real_data(self, idx_0=None, idx_f=None):
        """Output relevant data for analysis as lists"""
        if idx_0 is None:
            idx_0 = 0
        if idx_f is None:
            idx_f = self.n - 1

        rel_dates = self.list_date[idx_0:idx_f]
        rel_cumvol = self.list_cumvol[idx_0:idx_f + 1]
        rel_eruptvol = self.list_eruptvol[idx_0:idx_f]

        return rel_dates, rel_eruptvol, rel_cumvol

    def output_next(self, idx_next):
        rel_dates = self.list_date[idx_next]
        rel_cumvol = self.list_cumvol[idx_next + 1]
        rel_eruptvol = self.list_eruptvol[idx_next]

        return rel_dates, rel_eruptvol, rel_cumvol


    def piton_rates(self):
        """Set the rates of eruptions for PitondelaFournaise
        Source: manuscript by Galetto (2025), Q unit: km3/yr
        Q unit for computation: m3/day"""

        # Transform km3/yr ro m3/day
        self.Q1 = bf.Qy_to_Qday(0.0107)
        self.Q2 = bf.Qy_to_Qday(0.0228)
        self.Qlong = bf.Qy_to_Qday(0.0024)


class OfficialPeriod:
    """Class to handle periods of interest for the volcano data"""

    def __init__(self, period_number: int):
        """Initialize the period with eruption dates, volumes and cumulative volumes"""
        # todo: start, end dates, eruptions and volume, and q
        # TODO volume

        self.number= period_number

        # date of first and last eruption
        self.date_t0: datetime.date
        self.date_tf: datetime.date

        # first and last eruption ID
        self.e0: int
        self.ef: int

        # rate for the period
        self.q: float
        self.q_yr: float

    def set_dates(self, datet0, datetf):
        """Set the start and end dates of the period"""
        if isinstance(datet0, datetime.date) and isinstance(datetf, datetime.date):
            self.date_t0 = datet0
            self.date_tf = datetf
        else:
            exit("Invalid date format. Use datetime.date objects.")

    def set_eIDs(self, eid_t0, eid_tf):
        """Set the first and last eruption IDs of the period"""
        if isinstance(eid_t0, int) and isinstance(eid_tf, int):
            self.e0 = eid_t0
            self.ef = eid_tf
        else:
            exit("Invalid eruption ID format. Use integers.")

    def set_q(self, q):
        """Set the rate for the period"""

        self.q = q
        self.q_yr = bf.Qday_to_Qy(q)

if __name__ == "__main__":
    # create an instance of the class for a volcano
    piton = VolcanoData(name='PitondelaFournaise_data', printing=True)
    # get data from the file
    piton.organize()
