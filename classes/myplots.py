"""Plots all in one place."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time

from classes.basicfun import Basicfun as bf
from classes.collectdata import VolcanoData
from classes.eruption import OneEruption


class MyPlots:
    def __init__(self, piton=True):
        self.name = "MyPlots"

        # volcano name
        self.volcano_name: str
        self.volcano_subname: str
        self.period: str

        # type data
        self.words = ["Eruptions",
                     "Volume",
                     "Interval",
                      "Cumulative Volume"]

        # PLOTSET 1
        self.width = 14
        self.height = 6

        self.yth = 0.05  # y-axis limit for volume in m3

        self.label_mean = "Mean = "
        self.label_median = "Median = "
        self.short = 6
        self.unit = [f"10$^{self.short}$ m$^3$",'days']

        # titles
        self.title_evol = "Eruption Events"
        self.title_intervals = "Intervals between Eruptions"
        self.title_cvol = "Cumulative Volume"
        self.title_error_hist = "Error Distribution"

        # histogram
        self.n_bins = 25
        self.title_hist = "Distribution"

        # fig settings
        self.label_vol = f"Volume ($m^3$)"
        self.label_freq = "Frequency"
        self.label_date = "End of Eruption (Date)"
        self.label_interval =f"$\Delta~T$"
        self.label_number = "Eruption Number"

        self.savepath = self.get_save_path()

        if piton:
            self.set_piton()

        # TODO font type and size

    def set_piton(self):
        """Set volcano name and subname for Piton de la Fournaise."""
        self.volcano_name = "Piton de la Fournaise"
        self.volcano_subname = "Period"

    def save_fig(self, fig, savename=None):
        """Save figure to the specified path."""
        if savename is None:
            savename = 'plot'
        full_path = os.path.join(self.savepath, f'{savename}.pdf')
        fig.savefig(full_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved: {full_path}")


    def plot_set01(self, vd: VolcanoData, plot_op=1, savename=None):
        """Plot eruption volumes (evol) or time intervals (dT) with histograms
        Now: Piton de la Fournaise"""

        # compute stats for plotting
        vd.compute_for_plotting()
        self.period = bf.format_period(vd.list_date[0], vd.list_date[-1])

        # TITLE
        suptitle = f"{self.volcano_name} {self.words[0]} \n{self.period}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # -------------------- PLOT 1 (LEFT)
        if plot_op == 1:
            xvalues = vd.list_date
            yvalues = vd.list_eruptvol
            # mean and median for eruptions
            mean_value = vd.mean_evol
            median_value = vd.median_evol
            # label for volume
            label_mean = f"{mean_value/10**self.short:.2f} {self.unit[0]}"
            label_median = f"{median_value/10**self.short:.2f} {self.unit[0]}"
            labely = self.label_vol
            labelx = self.label_date
            # titles
            title1 = self.title_evol
            title2 = f"Volume {self.title_hist}"
        else:
            xvalues = [i for i in range(1, len(vd.list_date))]
            yvalues = vd.intervals
            # mean and median for intervals
            mean_value = vd.mean_dT
            median_value = vd.median_dT
            # label for time
            label_mean = f"{mean_value:.0f} {self.unit[1]}"
            label_median = f"{median_value:.0f} {self.unit[1]}"
            labely = f"{self.label_interval} ({self.unit[1]})"
            labelx = self.label_number
            # titles
            title1 = self.title_intervals
            title2 = f"Time Interval {self.title_hist}"


        # PLOT ERUPTIONS (EVOL)
        ax[0].scatter(xvalues, yvalues, marker='x', color='b', label=f"{self.words[0]}")
        # mean
        ax[0].axhline(mean_value, color='m', linestyle='--',
                   label=f"{self.label_mean} {label_mean}")
        # median
        ax[0].axhline(median_value, color='k', linestyle='--',
                   label=f"{self.label_median} {label_median}")

        # plot title, labels and legend
        ax[0].set(title=title1, xlabel=labelx, ylabel=labely)
        ax[0].legend(frameon=False)

        # set limits for x and y axes
        ylim = max(yvalues) * self.yth  # threshold for y-axis
        ax[0].set(ylim=(min(yvalues) - 2*ylim, max(yvalues) + ylim))
        # ------------------------------------- PLOT 2 (RIGHT)

        # Plot histogram with KDE
        sns.histplot(yvalues, kde=True, ax=ax[1], color='b', bins=self.n_bins)
        # extra stats
        ax[1].axvline(mean_value, color='m', linestyle='--',
                      label=f"{self.label_mean} {label_mean}")
        ax[1].axvline(median_value, color='k', linestyle='--',
                      label=f"{self.label_median} {label_median}")

        # plot identifiers
        plt.title(title2)
        plt.xlabel(labely)
        plt.ylabel(self.label_freq)
        plt.legend(frameon=False)


        # show, save and close
        if savename is None:
            savename = 'evol'
        self.save_fig(fig, savename)
        #plt.show()


    def plot_set02(self, eruptions: list, savename=None):
        """Plot Cumulative Volume (CVOL) real and deterministic prediction, and error histogram"""

        self.period = bf.format_period(eruptions[0].date.real, eruptions[-1].date.real)

        suptitle = f"{self.volcano_name} {self.words[3]} \n{self.period}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # -------------------- PLOT 1 (LEFT)
        xvalues = [e.date.real for e in eruptions[1:]]
        yvalues_real = [e.cvol.real for e in eruptions[1:]]  # skip first eruption
        yvalues_det = [e.cvol.det.value for e in eruptions if e.cvol.det.value is not None]
        q = bf.Qday_to_Qy(eruptions[1].q_period)

        leg1 = f"Real"
        leg2 = f"Predicted, Q = {q:.4f} km3/yr"

        # real values
        ax[0].scatter(xvalues, yvalues_det, marker='x', color='k', label=leg1)
        # predicted values
        ax[0].scatter(xvalues, yvalues_real, marker='x', color='b', label=leg2)
        # plot title, labels and legend
        ax[0].set(title=self.title_cvol, xlabel=self.label_date, ylabel=self.label_vol)
        ax[0].legend(frameon=False)

        # set limits for x and y axes
        ylim = max(yvalues_real) * self.yth  # threshold for y-axis
        ax[0].set(ylim=(min(yvalues_real) - 2 * ylim, max(yvalues_real) + ylim))

        # ------------------------------------- PLOT 2 (RIGHT)
        # Plot histogram with KDE
        yvalues_error = [e.cvol.det.error for e in eruptions[1:] if e.cvol.det.error is not None]
        sns.histplot(yvalues_error, kde=True, ax=ax[1], color='b', bins=self.n_bins)

        # plot identifiers
        plt.title(self.title_error_hist)
        plt.xlabel(self.label_vol)
        plt.ylabel(self.label_freq)
        plt.legend(frameon=False)

        # show, save and close
        if savename is None:
            savename = 'cvol'
        self.save_fig(fig, savename)
        plt.show()

        return

    @staticmethod
    def get_save_path():
        current_dir = os.getcwd()
        # Get the parent directory > '../PycharmProjects/volcano'
        parent_dir = os.path.dirname(current_dir)
        # fix path for when running unit tests
        if '/plots' not in parent_dir:
            parent_dir += '/plots'

        save_path = os.path.join(parent_dir, parent_dir)

        return save_path


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
    def get_plot_title(name: str, what="evol"):
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


