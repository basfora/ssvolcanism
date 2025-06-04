"""Importing data from Excel files and saving them to plot"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import statistics

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

        # as lists
        self.list_date = []
        self.list_eruptvol = []
        self.list_cumvol = []

        # --------------------------------
        # for computing stuff
        self.timeline = [0]
        self.n = 0  # number of eruptions
        self.Q_long = None

        # stats
        self.list_Q = []  # list of rates of eruptions
        self.mean_Q = None
        self.std_Q = None

        self.list_dt = []  # list of time intervals between eruptions
        self.mean_dt = None
        self.std_dt = None

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

    def collect_from(self):
        """Return path to Excel file: current > parent > rawData"""

        # Get the current working directory
        self.current_dir = os.getcwd()
        # Get the parent directory
        parent_dir = os.path.dirname(self.current_dir)
        # Get the path to the rawData folder
        self.raw_data_path = os.path.join(parent_dir, self.raw_data_dir)
        # path to file
        self.path_to_file = os.path.join(self.raw_data_path, self.name)
        # add extension (.xlsx) if not present
        if ".xlsx" not in self.path_to_file:
            self.path_to_file += self.extension

        return self.path_to_file

    def import_data(self):
        """Import data from an Excel file and return as DataFrame"""

        # get paths and extensions
        self.collect_from()
        # Read the Excel file
        df = pd.read_excel(self.path_to_file, date_format='%d/%m/%Y')

        return df

    def get_data(self, r1=1, rend=74):
        """Get data from file and return it as a list
        default: PitondelaFournaise
        years: 1936 - March 1998"""
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
        self.list_cumvol = self.series_cumvol.tolist()
        self.list_date = self.series_date.tolist()
        # some other computation (might delete later)
        self.compute_intervals_eruption()
        self.compute_timeline()

        if self.printing:
            # all dataframe
            print('Volcano df ', self.df_volcano)
            print('Cum Vol ', self.list_cumvol)
            # dates
            print('dt in days: ', self.list_dt)

        # only need date and cumulative volume
        return self.list_date, self.list_cumvol, self.list_eruptvol

    def get_plot_title(self):
        if "Piton" in self.name:
            title = "Piton de la Fournaise"
            xlabel = "Collection Date (year)"
            ylabel = "Volume ($m^3$)"
            legend = ["Cumulative Volume", "Erupted Volume"]
        else:
            title = "Unknown"
            xlabel = "Collection Date (year)"
            ylabel = "Cumulative Volume ($m^3$)"
            legend = ["Cumulative Volume", "Erupted Volume"]

        return title, xlabel, ylabel, legend

    def plot_volume(self):
        """Plot the cumulative volume of the volcano and date the data was collected"""

        title, xlabel, ylabel, mylegend = self.get_plot_title()
        fig, ax = plt.subplots()


        # convert date to string for plotting
        xvalues = self.timeline
        yvalues = self.list_cumvol

        # PLOT
        # cumulative volume
        ax.plot(xvalues, yvalues, linestyle='dashed', color='b', linewidth=0.5, marker='.', markersize=10, mfc='red', label=mylegend[0])
        # erupted volume
        yvalues2 = self.list_eruptvol
        ax.plot(xvalues, yvalues2 , linestyle='dashed', color='g', linewidth=0.5, marker='.', markersize=10,
                mfc='red', label=mylegend[1])

        # set limits for x and y axes
        th = 1e8
        ax.set(ylim=(min(yvalues) - th, max(yvalues) + th))
        # title and labels
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        # grid
        ax.grid()
        ax.set_xticks(xvalues[::1])
        ax.set_xticklabels(xvalues[::1], rotation=45, ha='right', fontsize=8)

        # show
        plt.show()
        # save the figure
        save_path = os.path.join(self.current_dir, self.plot_dir)
        fig.savefig(save_path + "/piton-cumvol.png")

    def compute_intervals_eruption(self):
        """Compute intervals between eruptions"""
        # convert date to datetime
        mydates = self.list_date
        # compute time between measurements
        for i in range(1, len(mydates)):
            date1 = mydates[i - 1]
            date2 = mydates[i]
            dt = self.compute_dt(date1, date2)
            self.list_dt.append(dt)

        if self.printing:
            print("Time between measurements (days):", self.list_dt)

    def compute_timeline(self):
        """Compute timeline as:
        first eruption = day 1 (default 1)
        next eruption = first eruption + dt """

        timeline = [0]
        for i in range(self.n-1):
            dt = self.list_dt[i]
            timeline.append(timeline[-1] + dt)
        self.timeline = timeline

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

    def add_rate_interval(self, Q_idx, dt_idx):
        """Add rate of eruptions to the list
        :param Q_idx: rate of eruptions in km3/yr
        :param dt_idx: time interval between eruptions in days"""
        self.list_Q.append(Q_idx)
        self.list_dt.append(dt_idx)

    def analyze_Q(self):
        """Analyze the rate of eruptions"""
        if len(self.list_Q) > 0:
            self.mean_Q = sum(self.list_Q) / len(self.list_Q)
            self.std_Q = statistics.stdev(self.list_Q)

            # self.all_dt = (self.list_date[-1] - self.list_date[0]).days/self.n  # total time of eruptions
            self.mean_dt = sum(self.list_dt) / len(self.list_dt)
            self.std_dt = statistics.stdev(self.list_dt)
        else:
            print("No rates to analyze.")

        if self.printing:
            self.print_rate_stats()

    def print_rate_stats(self):

        time_period_days = (self.list_date[-1] - self.list_date[0]).days
        time_period_years = self.list_date[-1].year - self.list_date[0].year  # approximate conversion to years
        # sanity check
        print(f"Period analyzed: {self.list_date[0]} - {self.list_date[-1]}")
        print(f"Time period: {time_period_days} days ({time_period_years} years)")
        print(f"Number of eruptions: {self.n}")
        print("...Time between between eruptions (dt in days):")
        print(f"From all period: tf - t0 / n \n> dt_all = {time_period_days / self.n:.2f} days")
        print(f"Mean adjusted with each new eruption\n> dt_iter = {self.mean_dt:.2f} +- {self.std_dt:.2f} days")

        print(f"...Rate of eruptions (Q in km3/year):")
        if self.Q_long is not None:
            print(f"Long-term: Q_long = {self.Q_long:.4f} km3/year")
        else:
            print("Long-term rate of eruptions not set.")

        print(f"From all period: [Vcum(tf) - Vcum(t0)] / dT \n>> Q_all = {self.list_Q[-1]:.4f} km3/year")
        print(f"Mean of Q adjusted with each new eruption \n>> Q_iter = {self.mean_Q:.5f} +- {self.std_Q:.5f} km3/year")

    def set_long_term_rate(self, Q_long_term: float):
        """Set the long-term rate of eruptions
        as computed in previous studies, in km3/yr """
        self.Q_long = Q_long_term


    @staticmethod
    def date_to_str(df_date):
        """Convert date to string format YY
        :return list of strings"""
        list_date = []

        for obj in df_date:
            # convert to datetime (Timestamp)
            obj = pd.to_datetime(obj)
            b = str(obj.month) + '.' + str(obj.year - 1900)  #+ "-" +  #+ "-" + str(obj.day)
            list_date.append(b)

        return list_date

    @staticmethod
    def compute_dt(date1, date2):
        """Compute time between two dates
        :param date1: first date
        :param date2: second date
        :return: time difference in days"""
        # convert to datetime
        date1 = pd.to_datetime(date1)
        date2 = pd.to_datetime(date2)

        dt = (date2 - date1).days

        return dt





if __name__ == "__main__":
    # create an instance of the class for a volcano
    piton = VolcanoData(name='PitondelaFournaise_data', printing=True)
    # get data from the file
    piton.get_data()
    # plot it (simple)
    piton.plot_volume()