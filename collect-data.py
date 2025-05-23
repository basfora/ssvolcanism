"""Importing data from excel files and saving them to plot"""

import pandas as pd
import os
import matplotlib.pyplot as plt


class VolData:
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

        # --------------------------------
        # DATA HANDLING
        # --------------------------------
        # original data frames
        self.df_volcano = None
        self.df_date = None
        self.df_eruptvol = None
        self.df_cumvol = None

        # as lists
        self.list_date = []
        self.list_eruptvol = []
        self.list_cumvol = []

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
        # parent_dir = os.path.dirname(current_dir)
        # Get the path to the rawData folder
        self.raw_data_path = os.path.join(self.current_dir, self.raw_data_dir)
        # path to file
        self.path_to_file = os.path.join(self.raw_data_path, self.name)
        # add extension (.xlsx) if not present
        if ".xlsx" not in self.path_to_file:
            self.path_to_file += self.extension

        return self.path_to_file

    def import_data(self):
        """Import data from an Excel file and return a DataFrame"""

        # get paths and extensions
        self.collect_from()
        # Read the Excel file
        df = pd.read_excel(self.path_to_file)

        return df

    def get_data(self):
        """Get data from file and return it as a list
        TODO: make limits modifiable,
        default: PitondelaFournaise
        years: 1936 - March 1998"""
        # ----------------------------------
        # columns to use: B and E
        cDate, cErupted, cCV = 1, 3, 4
        # rows to use: 1 - end
        r1, rend = 1, 74
        # ----------------------------------
        # import data from the file
        self.df_volcano = self.import_data()
        # -----------------------------------
        # get data from the file
        self.df_date = self.df_volcano.iloc[r1:rend, cDate]
        self.df_eruptvol = self.df_volcano.iloc[r1:rend, cErupted]
        self.df_cumvol = self.df_volcano.iloc[r1:rend, cCV]
        # -----------------------------------
        # turn into lists to make it easier to see/plot
        self.list_eruptvol = self.df_eruptvol.tolist()
        self.list_cumvol = self.df_cumvol.tolist()
        # todo: date may need to convert to datetime to get time between measurements
        self.list_date = self.df_date.tolist()

        if self.printing:
            # all dataframe
            print(self.df_volcano)
            print(self.list_cumvol)
            # dates
            # print(self.df_date)

        # only need date and cumulative volume
        return self.list_date, self.list_cumvol, self.list_eruptvol

    def get_plot_title(self):
        if "Piton" in self.name:
            title = "Piton de la Fournaise"
            xlabel = "Collection Date (year)"
            ylabel = "Cumulative Volume ($m^3$)"
        else:
            title = "Unknown"
            xlabel = "Collection Date (year)"
            ylabel = "Cumulative Volume ($m^3$)"

        return title, xlabel, ylabel

    def plot_volume(self):
        """Plot the cumulative volume of the volcano and date the data was collected"""

        fig, ax = plt.subplots()

        # convert date to string for plotting
        xvalues = self.date_to_str(self.df_date)
        yvalues = self.list_cumvol

        # PLOT
        ax.plot(xvalues, yvalues, linestyle='dashed', color='b', linewidth=0.5, marker='.', markersize=10, mfc='red')

        # set limits for x and y axes
        th = 1e8
        ax.set(ylim=(min(yvalues) - th, max(yvalues) + th))
        # title and labels
        title, xlabel, ylabel = self.get_plot_title()
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        # grid
        ax.grid()
        ax.set_xticks(xvalues[::1])
        ax.set_xticklabels(xvalues[::1], rotation=45, ha='right', fontsize=8)

        # show
        plt.show()
        # save the figure
        save_path = os.path.join(self.current_dir, self.plot_dir)
        fig.savefig(save_path + "/piton-cumvol.png")

    @staticmethod
    def date_to_str(df_date):
        """Convert date to string format YY
        :return list of strings"""
        list_date = []

        for obj in df_date:
            # convert to datetime (Timestamp)
            obj = pd.to_datetime(obj)
            b = str(obj.year - 1900)  # + "-" + str(obj.month) #+ "-" + str(obj.day)
            list_date.append(b)

        return list_date

if __name__ == "__main__":
    # create an instance of the class for a volcano
    piton = VolData(name='PitondelaFournaise_data', printing=True)
    # get data from the file
    piton.get_data()
    # plot it (simple)
    piton.plot_volume()