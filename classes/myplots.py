"""Plots all in one place."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from classes.basicfun import Basicfun as bf
from classes.collectdata import VolcanoData


class MyPlots:
    def __init__(self):
        self.name = "MyPlots"


    @staticmethod
    def plot_evol(vd: VolcanoData):
        """Plot eruption volumes (evol)"""

        # PLOT PTS AND HISTOGRAM WITH RIGHT TITLE (PERIOD)
        # TODO THIS FIRST THING AND PASTE TO OVERLEAF
        # ADD MEAN, MEDIAN and MODE (skewed right normal distribution)

        return


    @staticmethod
    def plot_eruptions(dates, volumes, cumulative_volumes):
        """Plot eruption volumes over time."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=dates, y=volumes, label='Eruption Volume (m3)', marker='o')
        sns.lineplot(x=dates, y=cumulative_volumes, label='Cumulative Volume (m3)', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Volume (m3)')
        plt.title('Eruption Volumes Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_data(dates: list, values: list, cvalues=None):

        """Plot data with histograms and time series."""

        plt.figure(figsize=(12, 12))
        nplots = 2

        # VOLUME ERUPTIONS (EVOL)
        n = len(dates)
        # mean and std
        mean_v, std_v = bf.compute_mean_std(values)
        mean_values = [mean_v] * n

        print(f"Number of eruptions: {n}")
        t0, tf = bf.print_period(dates[0], dates[-1])
        print(f"Volume per eruption (m3):\n Mean: {mean_v:.0f} \n Std Dev: {std_v:.0f}")

        if cvalues is not None:
            q = bf.compute_q(cvalues[0], cvalues[-1], (dates[-1] - dates[0]).days)
            e_line = [q * (dates[i] - dates[i-1]).days for i in range(1, n)]

        # eruptions and volumes
        title1 = 'Eruptions\n' + t0 + ' to ' + tf
        label1 = 'Volume ($m^3$)'
        title2 = 'Volume '

        plt.subplot(nplots, nplots, 1)
        plt.scatter(dates, values, marker='x', color='b', label='real data')
        plt.plot(dates, mean_values, color='k', linestyle='--')
        if cvalues is not None:
            #plt.plot(dates[1:], e_line, color='g', linestyle='--', label='Q line')
            plt.scatter(dates[1:], e_line, marker='x', color='r', label='Q * $\Delta T_{k}$')
            plt.legend()
        plt.title(title1)
        plt.xlabel("Date of Eruption")
        plt.ylabel(label1)

        plt.subplot(nplots, nplots, 2)
        sns.histplot(values, kde=True)
        plt.title("Distribution " + title2)
        plt.xlabel(label1)
        plt.ylabel("Frequency")

        # TIME INTERVALS (DT)
        values_dt_days = bf.compute_intervals(dates)
        aux_x = [i for i in range(n - 1)]
        title3 = 'Time Intervals '
        xlabel3 = 'Eruption Number'
        label3 = '$\Delta~T$ (days)'

        # mean and std
        mean_dt, std_dt = bf.compute_mean_std(values_dt_days)
        mean_dt_values = [mean_dt] * (n-1)

        print(f"Time interval between eruptions (days): \n Mean: {mean_dt:.0f} \n Std Dev: {std_dt:.0f}")

        plt.subplot(nplots, nplots, 3)
        plt.scatter(aux_x, values_dt_days, marker='x', color='b')
        plt.plot(aux_x, mean_dt_values, color='k', linestyle='--', label='Mean Interval')
        plt.title(title3)
        plt.xlabel(xlabel3)
        plt.ylabel(label3)

        plt.subplot(nplots, nplots, 4)
        sns.histplot(values_dt_days, kde=True)
        plt.title("Distribution " + title3)
        plt.xlabel(label3)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

        return values_dt_days


    @staticmethod
    def plot_cvol(dates: list, values: list, cvalues: list):
        """Plot cumulative volume with a rate line."""

        plt.figure(figsize=(12, 12))

        q = bf.compute_q(cvalues[0], cvalues[-1], (dates[-1] - dates[0]).days)
        q_years = bf.Qday_to_Qy(q)
        print(f"Computed rate Q = {q:.4f} m3/day ({q_years:.5f} km3/year)")
        cvol_theory = bf.get_q_line(q, dates, cvalues)

        values_dt_days = bf.compute_intervals(dates)

        # CUMULATIVE VOLUME ERUPTIONS (CVOL)
        n = len(dates)

        nrow = 2
        ncol = 2

        t0, tf = bf.print_period(dates[0], dates[-1])
        title1 = 'Cumulative Volume'
        title2 = '\n' + t0 + ' to ' + tf
        label1 = "Volume ($m^3$)"
        label3 = "Error ($m^3$)"

        # FIT TO LINE
        timeline = bf.compute_timeline(values_dt_days)

        # linear squares fit
        a, b = np.polyfit(timeline, cvalues, 1)
        print(f"Linear fit: y = {a:.4f}x + {b:.4f}")
        fit_values = [a * t + b for t in timeline]

        # using computed q
        q_pred_values = [q * t + cvalues[0] for t in timeline]

        # CVOL PLOT
        plt.subplot(nrow, ncol, 1)
        myleg = 'Q = ' + str(round(q_years, 4)) +' m3/day'
        plt.scatter(dates, cvol_theory, marker='x', color='r', label=myleg)
        plt.scatter(dates, cvalues, marker='x', color='b')
        plt.title(title1 + title2)
        plt.xlabel("Date of Eruption")
        plt.ylabel(label1)
        plt.legend()

        # dt vs evol
        # plt.scatter(values_dt_days, values[1:], marker='x', color='r')
        # plt.title("Eruption Volume vs Time Interval ")
        # plt.xlabel('$\Delta T$ (days)')
        # plt.ylabel(label1)

        plt.subplot(nrow, ncol, 2)
        plt.scatter(timeline, cvol_theory, marker='x', color='r', label='$Q * \Delta T_{k} + Cvol_{k-1}$')
        plt.scatter(timeline, cvalues, marker='x', color='b', label='real data')
        plt.plot(timeline, fit_values, color='g', linestyle='--',
                 label='polyfit \n(a,b) = ({:.0f}, {:.0f})'.format(a, b))
        # plt.scatter(timeline, q_pred_values, marker='.', color='k', label='q_predicted')
        plt.title(title1)
        plt.xlabel('Timeline (days)')
        plt.ylabel(label1)
        plt.legend()

        # ERROR BETWEEN CVOL_THEORY AND CVOL_DATA
        q_error = [cvol_theory[i-1] - cvalues[i] for i in range(1, n)]
        title3 = 'Error '
        aux_x = [i for i in range(n - 1)]
        error_mean, error_std = bf.compute_mean_std(q_error)
        error_mean_values = [error_mean] * (n - 1)
        print(f"Error Cvol theory and data: {abs(error_mean):.4f} +- {error_std:.4f} m3")

        plt.subplot(nrow, ncol, 3)
        plt.scatter(dates[1:], q_error, marker='o', color='b', label='CVol Error')
        plt.plot(dates[1:], error_mean_values, color='r', linestyle='--', label='Mean Error')
        plt.title(title3 + 'Cumulative Volume: Theory vs Data')
        plt.xlabel("Date of Eruption")
        plt.ylabel(label3)
        plt.legend()

        plt.subplot(nrow, ncol, 4)
        sns.histplot(q_error, kde=True)
        plt.title("Distribution " + title3)
        plt.xlabel(label3)
        plt.ylabel("Frequency")


        plt.tight_layout()
        plt.show()

        return q, timeline


    @staticmethod
    def get_plot_title(name: str):
        if "Piton" in name:
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
