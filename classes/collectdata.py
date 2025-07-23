"""Importing all data from Excel files and saving them
Return 3 lists: edates, evol, cvol"""
import numpy as np
import pandas as pd
import os
import copy

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

        #---------------------------------
        # source file is formatted
        #---------------------------------
        # TODO add here new info to expand to all volcanoes
        self.columns = {}
        self.n_periods = 0
        self.periods = {}

        # --------------------------------
        # PATHS
        # --------------------------------
        # current directory
        self.current_dir = ''
        # path to the raw data folder (where Excel file is)
        self.raw_data_path = ''
        self.path_to_file = ''

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
        self.edates_list = []
        self.evols_list = []
        self.cvols_list = []
        self.n = 0  # number of eruptions

        # --------------------------------
        # TODO DELETE ALL
        self.intervals = []  # list of intervals between eruptions
        self.timeline = []  # timeline of eruptions
        self.line_points = None  # points for linear extrapolation
        # --------------------------------
        # INIT FUNCTIONS
        # --------------------------------
        self.define_columns()

        self.printstatus(0)
        self.set_volcano_name(name)
        self.import_data()
        self.dataframe_to_lists()  # convert DataFrame to lists
        self.define_periods()      # organize into the official periods listed on file
        self.printstatus(-1)

        # ---------------------------------

    def define_columns(self):
        """Where to find the data, change if source file formatting changes"""

        # period parameters
        self.columns['period'] = 6  # column with period number
        self.columns['ID0'] = 7
        self.columns['IDf'] = 8
        self.columns['date_t0'] = 9
        self.columns['date_tf'] = 10
        self.columns['Q'] = 11  # rate in km3/yr

        # actual data
        self.columns['eID'] = 0  # eruption ID
        self.columns['date'] = 1
        self.columns['evol'] = 3
        self.columns['cvol'] = 4

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

        print(f"... Importing data of Volcano {self.volcano_name} from file: {self.name}")
        return self.path_to_file

    # UT - OK
    def import_data(self):
        """Import data from an Excel file and return as DataFrame"""

        # get paths and extensions
        self.collect_from()

        # Read the Excel file (all of it)
        self.df_volcano = pd.read_excel(self.path_to_file, date_format='%d/%m/%Y')

        return self.df_volcano

    def dataframe_to_lists(self):
        """Go into data frame and return lists of dates, eruption volumes and cumulative volumes"""

        # columns to use: B and E
        cID, cDate, cErupted, cCV = self.columns['eID'], self.columns['date'], self.columns['evol'], self.columns['cvol']

        # separate relevant data from the file, as DataFrames
        r1 = 1  # start from the second row (first row is header) until NaN
        self.series_eIDS = self.df_volcano.iloc[r1:, 0]  # eruption IDs
        self.series_date = self.df_volcano.iloc[r1:, cDate]
        self.series_eruptvol = self.df_volcano.iloc[r1:, cErupted]
        self.series_cumvol = self.df_volcano.iloc[r1:, cCV]

        # organize data into lists
        self.eIDs = self.series_eIDS.tolist()  # eruption IDs
        self.edates_list = [d.date() for d in self.series_date]
        self.evols_list = self.series_eruptvol.tolist()
        # adjust for fake zero init of CVOL (where it started to measure)
        self.cvols_list = [0.0] + self.series_cumvol.tolist()
        # number of eruptions
        self.n = len(self.evols_list)

        # print status
        print(f"... Imported data of {self.n} eruptions from {self.edates_list[0]} to {self.edates_list[-1]}")

    # OK
    def define_periods(self):
        """Extract parameters from the DataFrame for the volcano"""

        # columns of interest - Periods info
        cperiod = self.columns['period']
        cID0, cIDf = self.columns['ID0'], self.columns['IDf']
        cedatet0, cedatetf = self.columns['date_t0'], self.columns['date_tf']
        cQ = self.columns['Q']

        # less to write
        mydf = self.df_volcano
        row = 1
        while True:
            iperiod = mydf.iat[row, cperiod]  # period number
            if pd.isna(iperiod):
                break
            # get paramters for the period
            datet0, datetf = mydf.iat[row, cedatet0], mydf.iat[row, cedatetf]
            eID0, eIDf = mydf.iat[row, cID0], mydf.iat[row, cIDf]
            qyr = mydf.iat[row, cQ]

            # get cumulative volume at t0 (m3)
            if eID0 == 1:
                # first eruption, cumulative volume is 0
                cvolt0 = 0.0
            else:
                cvolt0 = mydf.iat[eID0-1, 4]
            # get cumulative volume at tf (m3)
            cvoltf = mydf.iat[eIDf, 4]

            # create period instance
            myperiod = MySubset(iperiod)
            # add period parameters
            myperiod.set_vname(self.volcano_name)
            myperiod.set_dates(datet0.date(), datetf.date())
            myperiod.set_eIDs(eID0, eIDf)
            myperiod.set_cvol(cvolt0, cvoltf)
            myperiod.set_q(qyr)
            # add actual data to the period instance
            edates, evols, cvols = self.select_data(eID0, eIDf)
            myperiod.set_lists(edates, evols, cvols)

            # store and print
            self.periods[myperiod.label] = myperiod
            row += 1

            # --- print info about the period ---
            # print(self.df_volcano)
            print(f'Saved Period {myperiod.label}: {myperiod.date_t0} - {myperiod.date_tf} (eruptions {myperiod.e0} - {myperiod.ef})', end=' ')
            print(f"Rate: {myperiod.qyr:.4f} km3/yr, cvol(t0): {myperiod.cvol_t0} | cvol(tf): {myperiod.cvol_tf} m3")

        # number of periods (do not change after this)
        self.n_periods = max(self.periods.keys())
        self.period_zero()  # create period 0 with all data

    # OK
    def period_zero(self):
        """Period 0 is all data (periods 1 and 2 combined)"""

        if self.n_periods == 1:
            # if there is only one period, just copy it
            myperiod = copy.deepcopy(self.periods[1])
        else:
            # create a new period 0
            myperiod = MySubset(0)
            lastpi = self.n_periods
            # t0: first period, tf: last period
            datet0, datetf = self.periods[1].date_t0, self.periods[lastpi].date_tf
            eID0, eIDf = self.periods[1].e0, self.periods[lastpi].ef
            cvolt0, cvoltf = self.periods[1].cvol_t0, self.periods[lastpi].cvol_tf

            # add period parameters
            myperiod.set_vname(self.volcano_name)
            myperiod.set_dates(datet0, datetf)
            myperiod.set_eIDs(eID0, eIDf)
            myperiod.set_cvol(cvolt0, cvoltf)
            # add actual data to the period instance
            edates, evols, cvols = self.select_data(eID0, eIDf)
            myperiod.set_lists(edates, evols, cvols)

        self.periods[0] = myperiod

        return

    # TODO UNIT TEST FOR NONE and ONE INSTANCE inputs
    def select_data(self, id0=None, idf=None):
        """Select data based on eruption IDs
        id0: first eruption ID, to start from first available eruption, use None
        idf: last eruption ID, to end at last available eruption, use None
        for only one eruption, use id0 = idf"""

        # for all available data, use None
        if id0 is None:  # get from first eruption
            id0 = 1
        if idf is None:  # get until last eruption
            idf = self.n

        # python uses 0-based indexing, so we need to adjust
        idx0, idxf = id0 - 1, idf - 1

        # if only one eruption is needed
        if id0 == idf:
            # return only one eruption
            edates = [self.edates_list[idx0]]
            evol = [self.evols_list[idx0]]
            cvol = [self.cvols_list[idx0 + 1]]
        else:
            edates = self.edates_list[idx0:idxf+1]
            evol = self.evols_list[idx0:idxf+1]
            cvol = self.cvols_list[idx0:idxf + 2]  # cvol starts with 0, so we need to include idf + 1

        return edates, evol, cvol

    @staticmethod
    def printstatus(opt=0):
        if opt == 0:
            print(f"==============================\n... Initializing VolcanoData collection")
        elif opt == -1:
            print(f"Data collection completed\n==============================")

    # todo move to prediction ----
    def linear_extrapolation(self, opt=1):
        """Linear extrapolation of eruption volumes and cumulative volumes"""

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


if __name__ == "__main__":
    """Main function to test the VolcanoData class"""
    vname = 'Piton'
    name_file = f'Table{vname}'
    # create an instance of the class for a volcano
    piton_data = VolcanoData(name_file, printing=True)

