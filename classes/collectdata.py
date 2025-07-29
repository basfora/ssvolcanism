"""Importing all data from Excel files and saving them
Return 3 lists: edates, evol, cvol"""
import numpy as np
import pandas as pd
import os
import copy

from classes.basicfun import Basicfun as bf
from classes.subset import MySubset
from classes.eruption import OneEruption


class VolcanoData:
    """Class to import data from excel files"""

    def __init__(self, name: str, printing=False):

        self.printing = printing
        self.save_plot = True
        # file name to be imported
        self.file_name = name
        self.extension = '.xlsx'
        self.raw_data_dir = 'rawdata'

        # --------------------------------
        # PATHS
        # --------------------------------
        # current directory
        self.current_dir = ''
        # path to the raw data folder (where Excel file is)
        self.raw_data_path = ''
        self.path_to_file = ''
        self.file_columns = {}  # source file is formatted
        self.define_columns()

        # --------------------------------
        # DATA HANDLING
        # --------------------------------
        self.volcano_name: str
        self.n_periods = 0
        # dictionary of MySubset instances, key is period number
        self.periods = {}
        # eruptions
        self.eruptions = list()  # list of OneEruption instances

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
        # INIT FUNCTIONS
        # --------------------------------
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
        self.file_columns['period'] = 6  # column with period number
        self.file_columns['ID0'] = 7
        self.file_columns['IDf'] = 8
        self.file_columns['date_t0'] = 9
        self.file_columns['date_tf'] = 10
        self.file_columns['Q'] = 11  # rate in km3/yr

        # actual data
        self.file_columns['eID'] = 0  # eruption ID
        self.file_columns['date'] = 1
        self.file_columns['evol'] = 3
        self.file_columns['cvol'] = 4

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
        self.path_to_file = os.path.join(self.raw_data_path, self.file_name)
        # add extension (.xlsx) if not present
        if ".xlsx" not in self.path_to_file:
            self.path_to_file += self.extension

        print(f"... Importing data of Volcano {self.volcano_name} from file: {self.file_name}")
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
        cID, cDate, cErupted, cCV = self.file_columns['eID'], self.file_columns['date'], self.file_columns['evol'], self.file_columns['cvol']

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
        cperiod = self.file_columns['period']
        cID0, cIDf = self.file_columns['ID0'], self.file_columns['IDf']
        cedatet0, cedatetf = self.file_columns['date_t0'], self.file_columns['date_tf']
        cQ = self.file_columns['Q']

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
            # compute Q-LINE for this period
            myperiod.compute_qline()

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
            lastpi = self.n_periods
            eID0, eIDf = self.periods[1].e0, self.periods[lastpi].ef

            # create a new period 0
            myperiod = self.create_subset(eID0, eIDf, 0)
            myperiod.compute_qline()

        self.periods[0] = myperiod

        return

    # UT - OK
    def select_data(self, id0=None, idf=None):
        """Select data based on eruption IDs
        id0: first eruption ID, to start from first available eruption, use -1
        idf: last eruption ID, to end at last available eruption, use -1
        for only one eruption, use id0: int, idf: None or id0
        for all data, use id0: None or -1, idf: None or -1

        Examples:
        # select data from eruptions id0 to idf (inclusive)
        vdata.select_data(id0, idf)
        # select data from eruption id0 (only)
        vdata.select_data(id0)
        # select data from start to eruption idf (inclusive)
        vdata.select_data(-1, idf)
        # select data from id0 to end (inclusive)
        vdata.select_data(id0, -1)
        # select data from start to end
        vdata.select_data(-1, -1) or vdata.select_data()

        Returns:
            edates: list of eruption dates
            evol: list of eruption volumes
            cvol: list of cumulative volumes
        """

        # return all data (both are None) or just one instance (id0 == idf)
        if idf is None:
            if id0 is None or id0 == -1:
                # if both are None, return all data
                id0, idf = -1, -1
            else:
                # id0 is valid, we want only one instance
                idf = id0

        # get from the start
        if id0 == -1:
            id0 = 1
        # get until the end
        if idf == -1:
            idf = self.n

        # python uses 0-based indexing, so we need to adjust
        idx0, idxf = id0 - 1, idf - 1

        # if only one eruption is needed
        if id0 == idf:
            # return only one eruption
            edates = self.edates_list[idx0]
            evol = self.evols_list[idx0]
            cvol = self.cvols_list[idx0 + 1]
        else:
            edates = self.edates_list[idx0:idxf + 1]
            evol = self.evols_list[idx0:idxf + 1]
            cvol = self.cvols_list[idx0:idxf + 2]  # cvol starts with 0, so we need to include idf + 1

        return edates, evol, cvol


    def create_subset(self, eID0=None, eIDf=None, label=-1):
        """Create a subset of the data based on eruption IDs (inclusive)"""

        # select data based on eruption IDs
        edates, evols, cvols = self.select_data(eID0, eIDf)

        if isinstance(evols, int) or isinstance(evols, float):
            # if only one eruption is selected, convert to lists
            edates = [edates]
            evols = [evols]
            cvols = [cvols]

        # easy access to key data
        datet0, datetf = edates[0], edates[-1]
        cvolt0, cvoltf = cvols[0], cvols[-1]

        # CREATE a new MySubset instance
        mysubset = MySubset(label)

        # add subset/period parameters
        mysubset.set_vname(self.volcano_name)
        mysubset.set_dates(datet0, datetf)
        mysubset.set_eIDs(eID0, eIDf)
        mysubset.set_cvol(cvolt0, cvoltf)

        # add actual data to the period instance
        mysubset.set_lists(edates, evols, cvols)

        return mysubset

    def create_eruptions_list(self, print_instance=False) -> dict:
        """Create an instance of OneEruption for each eruption in the data,
        Save in each instance:
         T2: real data for that eruption
         T1: and previous eruption
        PERIOD: period number, Q rate, Q-LINE parameters and results
        Output list of OneEruption instances

        Note: this can be done outside the class, but it is convenient to have it here
        """

        eruptions = dict()  # list to store OneEruption instances

        # create an instance for each eruption
        for i in range(self.n):
            eid = i + 1  # eruption ID starts from 1

            # create an instance of OneEruption with ID
            oe = OneEruption(eid)

            # T2: get and save data for the eruption (T2)
            edate, evol, cvol = self.select_data(eid)
            oe.save_real(edate, evol, cvol, 't2')

            # T1: get and save data for the previous eruption (T1)
            if eid > 1:
                edatet1, evolt1, cvolt1 = self.select_data(eid - 1)
            else:
                evolt1, cvolt1 = 0, 0   # first eruption has no previous eruption
                edatet1 = None         # use the same date for the first eruption
            oe.save_real(edatet1, evolt1, cvolt1, 't1') # save previous eruption data

            # PERIOD-RELATED DATA (Q-LINE)
            for pi in range(1, self.n_periods + 1):
                myperiod = self.periods[pi]

                # find which period this eruption belongs to
                if myperiod.e0 <= eid <= myperiod.ef:
                    # save period data in the eruption instance
                    oe.period = pi
                    # q rate of the period
                    oe.save_parameter(myperiod.q, 2)  # rate of eruption (m3/day)
                    # save q-line parameters and results to instance (q-line is period-based)
                    oe = myperiod.qline_method(oe)

                    break

            # save the eruption instance
            oe.print = print_instance  # save printing option
            eruptions[eid] = oe

        return eruptions

    @staticmethod
    def printstatus(opt=0):
        if opt == 0:
            print(f"==============================\n... Initializing VolcanoData collection")
        elif opt == -1:
            print(f"Data collection completed\n==============================")


if __name__ == "__main__":
    """Main function to test the VolcanoData class"""
    vname = 'Piton'
    name_file = f'Table{vname}'
    # create an instance of the class for a volcano
    piton_data = VolcanoData(name_file, printing=True)


