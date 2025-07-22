"""Importing all data from Excel files and saving them
Return 3 lists: edates, evol, cvol"""
import datetime
import numpy as np
import pandas as pd
import os
import copy

from classes.basicfun import Basicfun as bf
from classes.subset import MySubset


class VolcanoData:
    """Class to import data from excel files"""

    def __init__(self, name: str, printing=False):

        self.printing = printing
        self.save_plot = True
        # file name to be imported
        self.name = name
        self.extension = '.xlsx'
        self.raw_data_dir = 'rawdata'
        self.plot_dir = 'plots'

        #---------------------------------
        # source file is formatted
        #---------------------------------
        # TODO add here new info to expand to all volcanoes
        self.n_periods = 0
        self.periods = {}
        self.rlimits = {}  # row limits for each period, used for plotting
        self.columns = {}

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

        self.volcano_name: str

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
        self.set_volcano_name(name)
        self.import_data()
        self.extract_parameters()
        # ---------------------------------

    def define_columns(self):
        """Where to find the data, change if source file formatting changes"""


        return

    # OK
    def set_volcano_name(self, filename=None):
        """Set the name of the file to be imported"""

        """Set the name of the period"""
        if 'piton' in filename.lower():
            vname = 'Piton de la Fournaise'
        elif 'hawaii' in filename.lower():
            vname = 'Hawaii'
        elif 'iceland' in filename.lower():
            vname = 'Iceland'
        elif 'galapagos' in filename.lower():
            vname = 'Western Galapagos'
        else:
            vname = input("Enter the name of the volcano: ")

        self.volcano_name = vname
        print(f"Volcano: {self.volcano_name}")

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

        print(f"... Importing data from file: {self.name}")
        return self.path_to_file

    # UT - OK
    def import_data(self):
        """Import data from an Excel file and return as DataFrame"""

        # get paths and extensions
        self.collect_from()

        # Read the Excel file (all of it)
        self.df_volcano = pd.read_excel(self.path_to_file, date_format='%d/%m/%Y')

        return self.df_volcano

    # OK
    def extract_parameters(self):
        """Extract parameters from the DataFrame for the volcano"""

        # columns of interest - Periods info
        cperiod = 6
        cID0, cIDf = 7, 8
        cedatet0, cedatetf = 9, 10
        cQ = 11  # rate in km3/yr

        # less to write
        mydf = self.df_volcano
        row = 1
        while True:
            pi = mydf.iat[row, cperiod]  # period number
            if pd.isna(pi):
                break
            myperiod = MySubset(pi)
            # get paramters for the period
            datet0, datetf = mydf.iat[row, cedatet0], mydf.iat[row, cedatetf]
            eid_t0, eid_tf = mydf.iat[row, cID0], mydf.iat[row, cIDf]
            qyr = mydf.iat[row, cQ]

            # get volume in km3/yr
            if eid_t0 == 1:
                # first eruption, cumulative volume is 0
                cvol_t0 = 0.0
            else:
                # get cumulative volume at t0
                cvol_t0 = mydf.iat[eid_t0-1, 4]
            # get cumulative volume at tf
            cvol_tf = mydf.iat[eid_tf, 4]

            # save them in the period instance
            myperiod.set_dates(datet0.date(), datetf.date())
            myperiod.set_eIDs(eid_t0, eid_tf)
            myperiod.set_cvol(cvol_t0, cvol_tf)
            myperiod.set_q(qyr)

            # store and print
            self.periods[myperiod.label] = myperiod
            row += 1

            # --- print info about the period ---
            # print(self.df_volcano)
            print(f'Saved Period {myperiod.label}: {myperiod.date_t0} - {myperiod.date_tf} (eruptions {myperiod.e0} - {myperiod.ef})', end=' ')
            print(f"Rate: {myperiod.q_yr:.4f} km3/yr, cvol(t0): {myperiod.cvol_t0} | cvol(tf): {myperiod.cvol_tf} m3")

        # number of periods (do not change after this)
        self.n_periods = max(self.periods.keys())
        self.period_zero()  # create period 0 with all data

    # OK
    def period_zero(self):
        """Period 0 is all data (periods 1 and 2 combined)"""

        if self.n_periods == 1:
            # if there is only one period, just copy it
            p0 = copy.deepcopy(self.periods[1])
        else:
            # create a new period 0
            p0 = MySubset(0)
            lastkey = self.n_periods
            # t0: first period, tf: last period
            p0.set_dates(self.periods[1].date_t0, self.periods[lastkey].date_tf)
            p0.set_eIDs(self.periods[1].e0, self.periods[lastkey].ef)
            p0.set_cvol(self.periods[1].cvol_t0, self.periods[lastkey].cvol_tf)
            q = bf.compute_q(p0.cvol_t0, p0.cvol_tf, p0.date_dT)
            p0.set_q(q, 'day')  # set rate in m3/day

        self.periods[0] = p0

        return


    def organize_eruption_data(self, period=1):
        """Save eruption data into period instance"""
        # organize data for the period (todo merge with extract_parameters or organize period)
        list_date, list_eruptvol, list_cumvol = self.organize_period(period)

        myperiod = self.periods[period]

        # save data in the period instance
        myperiod.set_lists(list_date, list_eruptvol, list_cumvol)



    # TODO modify to integrate with subset
    def organize_period(self, period=1):
        """Get data from file and return it as lists
        :param period: 1 for period I, 2 for period II, 0 for all data
        """
        # ----------------------------------
        # todo merge with extract_parameters

        self.period = period
        if period > 0:
            pi = self.periods[period]
            r1, rend = pi.e0, pi.ef + 1
            self.date0 = pi.date_t0  # start date of the period
            cvolT0 = pi.cvol_t0
            # todo change that to be just q
            if pi.label == 1:
                self.Q1 = pi.q  # rate for period 1
            elif pi.label == 2:
                self.Q2 = pi.q  # rate for period 2
        else:   # todo dont need this anymore, use period 0
            r1 = self.periods[1].e0  # first eruption ID
            last_key = max(self.periods.keys())
            rend = self.periods[last_key].ef + 1    # first eruption ID + 1
            self.date0 = self.periods[1].date_t0
            cvolT0 = self.periods[1].cvol_t0

        # columns to use: B and E
        cDate, cErupted, cCV = 1, 3, 4
        # -----------------------------------

        # todo separate lists of ALL data (from import data)
        # todo break list of all data inside periods
        #  (add time intervals, timeline etc inside period instance)
        #  use period instances for q-fit
        #  use all data for deterministic and stochastic

        # TODO important commands start here!
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


    # todo MOVE TO SUBSET
    def compute_for_plotting(self):
        """Compute mean, median and mode for plotting"""



        # mean, median and mode for eruption intervals
        self.mean_dT, self.std_dT = bf.compute_mean_std(self.intervals)
        self.median_dT = bf.compute_median(self.intervals)


    # todo move to prediction ----
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
    # todo end --------------------------------


    # TODO modify to integrate with subset (use vd.period[0])
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


if __name__ == "__main__":
    # create an instance of the class for a volcano
    piton = VolcanoData(name='PitondelaFournaise_data', printing=True)
    # get data from the file
    piton.organize_period()
