"""Plots all in one place."""

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import seaborn as sns
import numpy as np
import os
import time
import datetime

from matplotlib.pyplot import scatter
from scipy.odr import odr_error

from classes.basicfun import Basicfun as bf
from classes.collectdata import VolcanoData
from classes.subset import MySubset
from classes.eruption import OneEruption

# todo ivestigate this warning, where is it coming from?
# os.environ['QT_QPA_PLATFORM'] = 'wayland'

class MyPlots:
    def __init__(self, name=None):

        # volcano name
        self.volcano_name = self.set_name(name)
        self.volcano_nickname = None

        self.volcano_subname: str
        self.period = None

        # PLOTSET 1
        self.width = 14
        self.height = 6
        # font sizes
        self.suptitle_fontsize = 11
        self.title_fontsize = 13
        self.leg_fontsize = 11
        self.label_fontsize = 11


        # fontdict
        self.suptitle_font = {'family': 'serif', 'weight': 'ultralight', 'size': self.suptitle_fontsize, }

        self.title_font = {'family': 'serif', 'color': 'black',
                           'weight': 'medium', 'size': self.title_fontsize,}

        self.label_font = {'family': 'serif', 'color': 'black',
                           'weight': 'light', 'size': self.label_fontsize,}

        self.leg_font = {'family': 'serif', 'weight': 'normal', 'size': self.leg_fontsize, }

        # 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
        # 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'

        self.yth = 0.07  # y-axis limit for volume in m3
        self.short = 6

        # type data
        self.w_volume = "Volume"
        self.w_interval = "Interval"
        self.w_cvol = "Cumulative Volume"
        self.w_eruptions = "Eruptions"

        self.label_break = ', '
        self.label_eruptions = "Eruptions" + self.label_break
        self.label_mean = "Mean" + self.label_break
        self.label_mean_symbol = "$\mu$ = "
        self.label_median = "Median" + self.label_break
        self.label_mode = "Mode" + self.label_break
        self.label_minmax = "Min/Max" + self.label_break
        self.label_real = "Real" + self.label_break
        self.unit = [f"10$^{self.short}$ m$^3$",'days']

        # titles
        self.title_break = ' - '
        self.title_events = "Eruption Events"
        self.title_intervals = "Intervals between Eruptions"
        self.title_cvol = "Cumulative Volume"
        self.title_evol = "Eruption Volume"
        self.title_error_hist = "Error Distribution"
        self.title_error = "Error"
        self.title_exp = "Real vs Expected"
        self.title_qline = 'Q-Line Approximation'
        self.title_det = 'Deterministic Time Interval'
        self.title_linear = "Linear Extrapolation"
        self.title_stochastic = "Stochastic Forecast"

        # legends
        self.leg_real = "Real Data"
        self.leg_pred = "Predicted Data"
        self.leg_exp = "Expected"
        self.leg_Q1 = "Period I, Q = "
        self.leg_Q2 = "Period II, Q = "
        self.leg_error = "Error"
        self.leg_ptcloud = "Simulated"
        self.leg_qhat = "$\hat{Q}$"
        self.leg_prior = 'Prior Data'

        # histogram
        self.n_bins = 20
        self.title_hist = "Distribution"

        # fig settings
        self.label_vol = f"Volume ($m^3$)"
        self.label_freq = "Frequency"
        self.label_date = "End of Eruption (Date)"
        self.label_interval =f"$\Delta~T (days)$"
        self.label_number = "Eruption Number"
        self.label_q = "km$^3$/yr"


        # COLORS, get from https://matplotlib.org/stable/gallery/color/named_colors.html
        self.color_real = 'navy'
        self.color_error = 'crimson' #'darkorange'
        self.color_mean = 'magenta' # 'darkgreen' # forestgreen'
        self.color_std = 'dimgray'
        self.color_median = 'darkgreen' #'goldenrod'
        self.color_mode = 'darkorange'  # 'darkgoldenrod'
        self.color_hist = 'cornflowerblue'
        self.color_pred1 = 'brown' # red
        self.color_pred2 = 'salmon'# 'rosybrown' # magenta
        self.color_ptcloud = 'darkgray'  # 'darkslategray'

        # linestyle
        self.line_mean = '-'
        self.line_median = '-'
        self.line_mode = '-'
        self.line_std = '--'
        self.line_linear = '-'
        self.line_real = '-'

        # marker
        self.marker_real = 'x'
        self.marker_det = 'x'
        self.marker_error = '.'
        self.marker_linear = '.'
        self.marker_ptcloud = '.'
        self.marker_mean = '.'
        self.marker_median = '.'

        # linewidth
        self.width_thick = 3
        self.width_med = 2
        self.width_thin = 1
        self.width_thinner = 0.5

        self.width_real = self.width_thick
        self.width_ptcloud = 0.5

        self.savepath = self.get_save_path()

    def set_name(self, name=None):
        """Set the name of the volcano"""

        if 'piton' in name.lower():
            self.volcano_name = 'Piton de la Fournaise'
        elif 'hawaii' in name.lower():
            self.volcano_name = 'Hawaii'
        elif 'iceland' in name.lower():
            self.volcano_name = 'Iceland'
        elif 'galapagos' in name.lower():
            self.volcano_name = 'Western Galapagos'
        else:
            self.volcano_name = input("Enter the name of the file: ")

        self.volcano_subname = "Period"

        return self.volcano_name

    def save_fig(self, fig, savename=None):
        """Save figure to the specified path."""
        if savename is None:
            savename = 'plot'
        full_path = os.path.join(self.savepath, f'{savename}.pdf')
        fig.savefig(full_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved: {full_path}")

    def plot_set01(self, vd: MySubset, plot_op=1, savename=None, plot_show=False):
        """Plot eruption volumes (evol) or time intervals (dT) with histograms"""

        self.period = bf.format_period(vd.date_t0, vd.date_tf)

        # TITLE
        suptitle = f"{self.volcano_name} {self.w_eruptions} \n{self.period}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # -------------------- PLOT 1 (LEFT)
        if plot_op == 1:    # eruption events (evol)
            xvalues = vd.edates
            yvalues = vd.evols
            # mean and median for eruptions
            mean_value = vd.evol_mean
            median_value = vd.evol_median
            std_value = vd.evol_std
            mode_value = vd.evol_mode
            label_std = f"{std_value/10**self.short:.2f} {self.unit[0]}"
            # label for volume
            label_eruptions = f"{vd.n}"
            label_mean = f"{mean_value/10**self.short:.2f} {self.unit[0]}"
            label_median = f"{median_value/10**self.short:.2f} {self.unit[0]}"
            label_mode = f"{mode_value/10**self.short:.2f} {self.unit[0]}"
            label_max = f"{min(yvalues)/10**self.short:.2f}/{max(yvalues)/10**self.short:.2f} {self.unit[0]}"
            # label_min = f"Min = {min(yvalues)/10**self.short:.2f} {self.unit[0]}"
            labely = self.label_vol
            labelx = self.label_date
            # titles
            title1 = self.title_events
            title2 = f"Volume {self.title_hist}"
        else:               # time intervals (dT)
            xvalues = [i for i in range(1, len(vd.edates))]
            yvalues = vd.intervals
            # mean and median for intervals
            mean_value = vd.dT_mean
            median_value = vd.dT_median
            std_value = vd.dT_std
            mode_value = vd.dT_mode
            label_std = f"{std_value:.0f} {self.unit[1]}"
            # label for time
            label_eruptions = f"{vd.n}"
            label_mean = f"{mean_value:.0f} {self.unit[1]}"
            label_median = f"{median_value:.0f} {self.unit[1]}"
            label_mode = f"{mode_value:.0f} {self.unit[1]}"
            label_max = f"{min(yvalues):.0f}/{max(yvalues):.0f} {self.unit[1]}"
            # label_min = f"Min = {min(yvalues):.0f} {self.unit[1]}"
            labely = f"{self.label_interval}"
            labelx = self.label_number
            # titles
            title1 = self.title_intervals
            title2 = f"Time Interval {self.title_hist}"




        # PLOT ERUPTIONS (EVOL)
        ax[0].scatter(xvalues, yvalues, marker=self.marker_real, color=self.color_real,
                      label=f'{self.label_eruptions}{label_eruptions}')

        # mean
        ax[0].axhline(mean_value, color=self.color_mean, linestyle=self.line_mean,
                   label=f"{self.label_mean}{label_mean}")
        # median
        ax[0].axhline(median_value, color=self.color_median, linestyle=self.line_mean,
                   label=f"{self.label_median}{label_median}")
        # mode
        ax[0].axhline(mode_value, color=self.color_mode, linestyle=self.line_mode,
                      label=f"{self.label_mode}{label_mode}")

        # extra stats (MAX and MIN)
        ymin, ymax = bf.compute_limits(yvalues)
        idx_min, idx_max = yvalues.index(ymin), yvalues.index(ymax)
        xmin, xmax = xvalues[idx_min], xvalues[idx_max]
        ax[0].scatter(xmin, ymin, marker='o', facecolors='none', edgecolors='red', s=100)
        ax[0].scatter(xmax, ymax, marker='o', facecolors='none', edgecolors='red', s=100, label=f'{self.label_minmax}{label_max}')


        # std
        #ax[0].axhline(mean_value + std_value, color=self.color_std, linestyle=self.line_std,
        #               label=f"$\pm \sigma$ = {label_std}")
        # ax[0].axhline(mean_value - std_value, color=self.color_std, linestyle=self.line_std)

        # plot title, labels and legend
        ax[0].set(title=title1, xlabel=labelx, ylabel=labely)
        ax[0].legend(frameon=False, fontsize=self.leg_fontsize, loc='upper left')

        # set limits for x and y axes
        ylim = max(yvalues) * self.yth  # threshold for y-axis
        ax[0].set(ylim=(min(yvalues) - 2*ylim, max(yvalues) + ylim))
        # ------------------------------------- PLOT 2 (RIGHT)

        # Plot histogram with KDE
        sns.histplot(yvalues, kde=True, ax=ax[1], color=self.color_hist, bins=self.n_bins)
        # extra stats
        ax[1].axvline(mean_value, color=self.color_mean, linestyle=self.line_mean,
                      label=f"{self.label_mean} {label_mean}")
        ax[1].axvline(median_value, color=self.color_median, linestyle=self.line_median,
                      label=f"{self.label_median} {label_median}")
        ax[1].axvline(mode_value, color=self.color_mode, linestyle=self.line_mode,
                       label=f"{self.label_mode} {label_mode}")

        # std
        #ax[1].axvline(mean_value + std_value, color=self.color_std, linestyle=self.line_std,
        #               label=f"$\pm \sigma$ = {label_std}")
        # ax[1].axvline(mean_value - std_value, color=self.color_std, linestyle=self.line_std)

        # plot identifiers
        plt.title(title2)
        plt.xlabel(labely)
        plt.ylabel(self.label_freq)
        plt.legend(frameon=False, fontsize=self.leg_fontsize)


        # show, save and close
        if savename is None:
            savename = 'evol'
        self.save_fig(fig, savename)
        if plot_show:
            plt.show()

    def plot_real_vs_expected(self, eruptions: list, option='cvol', method='qline', savename=None, show_plot=True):
        """Plot Volume (CVOL or EVOL) real and deterministic prediction
        :param eruptions: list of OneEruption instances
        :param option: 'cvol' for cumulative volume, 'evol' for eruption volume
        :param method: 'qline' for Q-line approximation, 'det' for deterministic prediction
        :param savename: name to save figure
        :param show_plot: whether to show the plot before saving it"""

        # get date limits of the plot for the title
        self.period = bf.format_period(eruptions[0].date.t2, eruptions[-1].date.t2)

        # supertitle: Volcano Name and Period (dates)
        suptitle = f"{self.volcano_name}{self.label_break}{self.period}"
        fig, ax = plt.subplots(1, 1, figsize=(self.width, self.height))
        fig.suptitle(suptitle, fontproperties=self.suptitle_font)

        # separate eruptions by periods
        ep1 = [oe for oe in eruptions if oe.period == 1]  # period I
        ep2 = [oe for oe in eruptions if oe.period == 2]  # period II

        # get Q values for periods
        q1 = bf.Qday_to_Qy(ep1[0].qperiod)  # Q for period I
        if len(ep2) > 0:
            q2 = bf.Qday_to_Qy(ep2[0].qperiod)  # Q for period I

        # sanity check for Piton de la Fournaise volcano
        if 'piton' in self.volcano_name.lower():
            self.sanity_piton_periods(ep1, ep2)

        # -------------------------------- DATA PREPARATION
        # xvalues: dates of eruptions
        xvalues = [e.date.t2 for e in eruptions]    # n
        xvalues1 = [e.date.t2 for e in ep1 if getattr(getattr(e, option), method).value is not None]         # Period I
        xvalues2 = [e.date.t2 for e in ep2 if getattr(getattr(e, option), method).value is not None]

        # yvalues: real cumulative volumes (CVOL)
        yvalues_real = [getattr(e, option).t2 for e in eruptions]  # n

        if option == 'cvol':
            myvol = f"{self.title_cvol}"

            # expected values (qline or det prediction)
            yvalues1 = [getattr(e.cvol, method).value for e in ep1 if getattr(e.cvol, method).value is not None]
            if len(ep2) > 0:
                yvalues2 = [getattr(e.cvol, method).value for e in ep2 if getattr(e.cvol, method).value is not None]

        elif option == 'evol':
            myvol = f"{self.title_evol}"

            # expected values (qline or det prediction)
            yvalues1 = [getattr(e.evol, method).value for e in ep1 if getattr(e.evol, method).value is not None]
            if len(ep2) > 0:
                yvalues2 = [getattr(e.evol, method).value for e in ep2 if getattr(e.evol, method).value is not None]

        else:
            raise "Option must be 'cvol' or 'evol'."

        # style qline or det
        if method == 'qline':
            mymarker = self.marker_linear
            mytitle = f"{myvol}{self.title_break}{self.title_qline}"
            mylinestyle = self.line_linear
        elif method == 'det':
            mymarker = self.marker_det
            mytitle = f"{myvol}{self.title_break}{self.title_det}"
            mylinestyle = 'None'
        else:
            raise "Method must be 'qline' or 'det'."

        # ------------------------------- PLOT 1 (MAIN)
        # PLOT REAL VALUES (n)
        ax.scatter(xvalues, yvalues_real, marker=self.marker_real, color=self.color_real, linewidth=3,
                   label=f"{self.leg_real}{self.label_break}{len(eruptions)} eruptions")

        # PLOT PREDICTED VALUES
        # period I
        ax.plot(xvalues1, yvalues1, marker=mymarker, color=self.color_pred1,
                linestyle=mylinestyle, linewidth=1, label=f"{self.leg_Q1}{q1:.4f} {self.label_q}")

        # if period II exists, plot predicted values
        if len(ep2) > 0:
            # Plot predicted values - period II (eruptions # 74 - end)
            ax.plot(xvalues2, yvalues2, marker=mymarker, color=self.color_pred2,
                    linestyle=mylinestyle, linewidth=1, label=f"{self.leg_Q2}{q2:.4f} {self.label_q}")

        # plot title, labels and legend
        ax.set_title(mytitle, fontdict=self.title_font)
        ax.set_xlabel(self.label_date, fontdict=self.label_font)
        ax.set_ylabel(self.label_vol, fontdict=self.label_font)
        ax.legend(frameon=False, prop=self.leg_font, loc='upper left')
        ax.grid(True)

        # set limits for x and y axes
        ylim = max(yvalues_real) * self.yth  # threshold for y-axis
        ax.set(ylim=(min(yvalues_real) - 2 * ylim, max(yvalues_real) + ylim))

        # show, save and close
        if show_plot:
            plt.show()
        if savename is None:
            # add volcano nickname to savename
            savename = f"{self.volcano_nickname}_{option}"
        self.save_fig(fig, savename)

    def plot_volume_error(self, eruptions: list, option='cvol', method='det', savename=None, show_plot=True):
        """Plot ERROR between real and predicted VOL, and error histogram"""

        # get data pts (option) and title (method)
        if option == 'cvol':
            myvol = self.title_cvol
            yvalues = [getattr(e.cvol, method).error for e in eruptions if getattr(e.cvol, method).value is not None]

        elif option == 'evol':
            myvol = self.title_evol
            yvalues = [getattr(e.evol, method).error for e in eruptions if getattr(e.evol, method).value is not None]

        else:
            exit()

        if method == 'linear':
            myvol += f" {self.title_linear}"

        if method == 'qline':
            myvol += f" {self.title_qline}"

        self.period = bf.format_period(eruptions[0].date.t2, eruptions[-1].date.t2)

        suptitle = f"{self.volcano_name}{self.label_break}{self.period}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle, fontproperties=self.suptitle_font)

        # ------------------------------------- PLOT 02: ERROR
        xvalues = [e.date.t2 for e in eruptions if getattr(e.cvol, method).value is not None]  # n-1

        # add stats: mean and std
        error_mean, error_std = bf.compute_mean_std(yvalues)

        # km3 for printing
        yvalues_km3 = bf.m3_to_km3(yvalues)
        # some error stats
        e_mean_km3, e_std_km3 = bf.compute_mean_std(yvalues_km3)
        e_median_km3 = bf.compute_median(yvalues_km3)
        e_var_km3 = bf.var_from_std(e_std_km3)
        e_min_km3, e_max_km3 = bf.compute_limits(yvalues_km3)
        e_mse_km3 = bf.compute_mse(yvalues_km3)
        e_rmse_km3 = bf.compute_rmse(yvalues_km3)

        # PLOT 1 (LEFT) - error pts and mean/std lines

        # Plot error between real and predicted CVOL
        ax[0].scatter(xvalues, yvalues, marker=self.marker_error, color=self.color_error, linewidth=2,
                      label=self.leg_error)
        # extra stats
        ax[0].axhline(error_mean, color=self.color_mean, linestyle=self.line_mean, linewidth=2,
                      label=f"{self.label_mean_symbol} {e_mean_km3:.4f} km$^3$")
        ax[0].axhline(error_mean + error_std, color=self.color_std, linestyle=self.line_std,
                      label=f"$\pm \sigma$ = {e_std_km3:.4f} km$^3$")
        ax[0].axhline(error_mean - error_std, color=self.color_std, linestyle=self.line_std)

        # median
        # ax[0].axhline(error_median, color=self.color_median, linestyle=self.line_mean,
        #              label=f"{self.label_median} km$^3$")

        # plot title, labels and legend
        ax[0].set_title(f"{myvol}{self.title_break}{self.title_error}", fontdict=self.title_font)
        ax[0].set_xlabel(self.label_date, fontdict=self.label_font)
        ax[0].set_ylabel(self.label_vol, fontdict=self.label_font)
        ax[0].legend(frameon=False, prop=self.leg_font)
        ax[0].grid(True)

        # set limits for x and y axes
        ylim = max(yvalues) * self.yth
        ax[0].set(ylim=(min(yvalues) - 2 * ylim, max(yvalues) + ylim))

        # ------------------------------------- PLOT 2 (RIGHT) - ERROR HISTOGRAM
        # Plot histogram with KDE
        sns.histplot(yvalues, kde=True, ax=ax[1], color=self.color_hist, bins=self.n_bins)

        # plot identifiers
        plt.title(self.title_error_hist, fontdict=self.title_font)
        # plot stats
        plt.axvline(error_mean, color=self.color_mean, linestyle=self.line_mean,
                    label=f"{self.label_mean_symbol} {e_mean_km3:.4f} km$^3$")
        plt.axvline(error_mean + error_std, color=self.color_std, linestyle=self.line_std,
                    label=f"$\pm \sigma$ = {e_std_km3:.4f} km$^3$")
        plt.axvline(error_mean - error_std, color=self.color_std, linestyle=self.line_std)

        plt.xlabel(self.label_vol, fontdict=self.label_font)
        plt.ylabel(self.label_freq, fontdict=self.label_font)
        plt.legend(frameon=False, prop=self.leg_font)

        # PRINTOUT error stats

        print(f"-----------")
        print(f"Error Statistics ({option} {method}):")
        print(f"Min: {e_min_km3:.6f} | Max: {e_max_km3:.6f} km3")
        print(f"Median: {e_median_km3:.4f} km3")
        print(f"Mean: {e_mean_km3:.4f} +- {e_std_km3:.4f} km3")
        print(f"Variance: {e_var_km3:.6f} km6")
        print(f"MSE: {e_mse_km3:.9f} km6")
        print(f"RMSE: {e_rmse_km3:.6f} km3")

        abs_error = [abs(e) for e in yvalues_km3]
        abs_e_min, abs_e_max = bf.compute_limits(abs_error)
        abs_e_mean, _ = bf.compute_mean_std(abs_error)
        abs_e_sum = np.sum(abs_error)
        print('Absolute Error Statistics:')
        print(f"Min: {abs_e_min:.4f} | Max: {abs_e_max:.4f} km3")
        print(f"Mean: {abs_e_mean:.6f} km3")
        print(f"Error Sum: {abs_e_sum:.6f} km3")
        # ----------------

        # show, save and close
        if show_plot:
            plt.show()
        if savename is None:
            savename = option
        self.save_fig(fig, savename)

        return

    def plot_stochastic(self, eruptions, eruption_id: int, savename=None, show_plot: bool = True):
        """Plot stochastic forecast for one eruption."""
        # Plot 6: Stochastic Forecast


        suptitle = f"{self.volcano_name} - Eruption #{eruption_id}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # eruption being estimated, get by the id
        oe = eruptions[eruption_id - 1]  # eruptions are 1-indexed, so id-1

        # PREP DATA
        # - real data
        xvalues_real = [e.date.t2 for e in eruptions if e.id < oe.id]
        yvalues_real = [e.cvol.t2 for e in eruptions if e.id < oe.id]
        # - stochastic forecast - point cloud
        xvalues_stoc = [oe.date.sim.pts[i].value for i in range(oe.cvol.sim.N)]
        yvalues_stoc = [oe.cvol.sim.pts[i].value for i in range(oe.cvol.sim.N)]

        # plot stats
        my_mean, my_median = oe.cvol.sim.mean.value, oe.cvol.sim.median.value
        my_std = oe.cvol.sim.std
        my_mode = oe.cvol.sim.mode.value


        date_mean = bf.transform_days_to_date(oe.dT.sim.mean.value, oe.date.t2)
        date_median = bf.transform_days_to_date(oe.dT.sim.median.value, oe.date.t2)
        date_mode = bf.transform_days_to_date(oe.dT.sim.mode.value, oe.date.t2)

        # prior eruptions (real, prior data)
        ax[0].scatter(xvalues_real, yvalues_real, marker=self.marker_real, color=self.color_real,
                      linewidth=self.width_med, label=self.leg_prior)

        # - stochastic forecast - point cloud (simulated data)
        leg_s = f'{self.leg_ptcloud} (N = {oe.cvol.sim.N})'
        ax[0].scatter(xvalues_stoc, yvalues_stoc, marker=self.marker_ptcloud, color=self.color_ptcloud, linewidth=self.width_ptcloud, label=leg_s)

        # plot mean and median lines
        ax[0].axhline(my_mean, color=self.color_mean, linestyle=self.line_mean,  linewidth=self.width_thinner)#, label=f"$\mu$")
        ax[0].axhline(my_median, color=self.color_median, linestyle=self.line_median,  linewidth=self.width_thinner)#, label=f"median")
        ax[0].axhline(my_mode, color=self.color_mode, linestyle=self.line_mode, linewidth=self.width_thinner)

        ax[0].axvline(date_mean, color=self.color_mean, linestyle=self.line_mean, linewidth=self.width_thinner)
        ax[0].axvline(date_median, color=self.color_median, linestyle=self.line_median, linewidth=self.width_thinner)
        ax[0].axvline(date_mode, color=self.color_mode, linestyle=self.line_mode, linewidth=self.width_thinner)

        # real eruption data
        ax[0].scatter(oe.date.t2, oe.cvol.t2, marker='*', color='orangered', linewidth=self.width_med,
                      label=self.leg_real)

        # plot title, labels and legend
        mytitle = f"{self.title_cvol}"
        ax[0].set(title=mytitle, xlabel=self.label_date, ylabel=self.label_vol)
        ax[0].legend(frameon=False, loc='center left')
        ax[0].grid(True)

        # PLOT 2 (RIGHT) - HISTOGRAM
        sns.histplot(yvalues_stoc, kde=True, ax=ax[1], color=self.color_hist, bins=self.n_bins)

        # STATS
        mean_km3, median_km3, mode_km3 = bf.m3_to_km3(my_mean), bf.m3_to_km3(my_median), bf.m3_to_km3(my_mode)
        std_km3 = bf.m3_to_km3(my_std)

        plt.axvline(my_mean + my_std, color=self.color_std, linestyle=self.line_std, linewidth=self.width_thinner,
                    label=f"$\pm \sigma$ = {std_km3:.4f} km$^3$")

        plt.axvline(my_mean - my_std, color=self.color_std, linestyle=self.line_std, linewidth=self.width_thinner)

        plt.axvline(my_mean, color=self.color_mean, linestyle=self.line_mean,
                    label=f"{self.label_mean} {mean_km3:.4f} km$^3$")

        plt.axvline(my_median, color=self.color_median, linestyle=self.line_median,
                    label=f"{self.label_median} {median_km3:.4f} km$^3$")

        plt.axvline(my_mode, color=self.color_mode, linestyle=self.line_mode,
                    label=f"{self.label_mode} {mode_km3:.4f} km$^3$")

        # REAL data
        real_km3 = bf.m3_to_km3(oe.cvol.t2)
        plt.axvline(oe.cvol.t2, color=self.color_real, linestyle=self.line_real,
                    label=f"{self.label_real} {real_km3:.4f} km$^3$")


        plt.title(self.title_hist)
        plt.xlabel(self.label_vol)
        plt.ylabel(self.label_freq)
        plt.legend(frameon=False)

        # show, save and close
        if show_plot:
            plt.show()
        if savename is None:
            savename = f'Piton_Sthocastic_{oe.id}'
        self.save_fig(fig, savename)


        return

    # wrapper for each method plots
    def det_plots(self, eruptions: list, show_plot: bool):

        # Plot 1: CVOL real vs expected (DET)
        base_name = f'{self.volcano_nickname}_Period0_Cvol_Det'
        self.plot_real_vs_expected(eruptions, 'cvol', 'det',
                                   base_name, show_plot)

        # Plot 2: CVOL error
        base_name = f'{self.volcano_nickname}_Period0_Cvol_DetError'
        self.plot_volume_error(eruptions, 'cvol', 'det',
                                   base_name, show_plot)

        # Plot 3: EVOL real vs expected
        base_name = f'{self.volcano_nickname}_Period0_Evol_Det'
        self.plot_real_vs_expected(eruptions, 'evol', 'det',
                                   base_name, show_plot)

        # Plot 4: EVOL error
        base_name = f'{self.volcano_nickname}_Period0_Evol_DetError'
        self.plot_volume_error(eruptions, 'evol', 'det',
                                   base_name, show_plot)

    def linear_plots(self, eruptions: list, show_plot: bool):
        # Plot 5: CVOL real vs expected (LINEAR)
        base_name = f'{self.volcano_nickname}_Period0_Cvol_QLine'
        self.plot_real_vs_expected(eruptions, 'cvol', 'qline',
                                   base_name, show_plot)

        base_name = f'{self.volcano_nickname}_Period0_Cvol_QLineError'
        self.plot_volume_error(eruptions, 'cvol', 'qline',
                                   base_name, show_plot)

    def stoc_plots(self, eruptions: list, eruptions_to_plot: list, show_plot: bool = True):
        """Plot stochastic forecast for all eruptions."""
        # Plot 6: Stochastic Forecast

        for eid in eruptions_to_plot:
            base_name = f'{self.volcano_nickname}_Stochastic_{eid}'
            self.plot_stochastic(eruptions, eid, base_name, show_plot)

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
    def sanity_piton_periods(ep1: list, ep2: list):

        # sanity check (date and q used)
        assert len(ep1) == 73, f"Period I should have 73 eruptions, it has {len(ep1)}"
        assert len(ep2) == 46, f"Period II should have 46 eruptions, it has {len(ep2)}"
        assert ep1[0].date.t2 == datetime.date(1936, 8, 1), "First date of period I should be 1936-1-8"
        assert ep1[-1].date.t2 == datetime.date(1998, 3, 11), "Last date of period I should be 1998-03-11"
        assert round(ep1[1].qperiod, 4) == round(ep1[-1].qperiod, 4) == round(bf.Qy_to_Qday(0.0107),
                                                                              4), "Q for period I should be 0.0107 km3/yr"

        assert ep2[0].date.t2 == datetime.date(1999, 7, 19), "First date of period II should be 1999-07-19"
        assert ep2[-1].date.t2 == datetime.date(2018, 7, 13), "First date of period II should be 2018-07-13"
        assert round(ep2[0].qperiod, 4) == round(ep2[-1].qperiod, 4) == round(bf.Qy_to_Qday(0.0228),
                                                                              4), "Q for period II should be 0.0228 km3/yr"


    @staticmethod
    def sanity_check_det(eruptions: list):
        """Print results of deterministic prediction."""
        print("Deterministic Prediction CVOL(T2) in km3:")

        print('ID ; DATE ; DT (days) ; CVOL REAL ; EXPECTED ; ERROR ; % ; Q (km3/yr) ; ')

        for e in eruptions:

            if e.id == 1 or e.id == 74:
                continue  # skip first eruption and first of period II

            # parameters for deterministic method
            dT = e.dT.t1_2
            e_previous = eruptions[e.id - 2]
            assert e_previous.cvol.t2 == e.cvol.t1, f"Cumulative volume at T1 does not match previous eruption's CVOL(T2) - {e.id}: {e_previous.cvol.t2, e.cvol.t1}"

            # check state equation and error
            cvolT2 = e.qperiod * dT + e.cvol.t1
            assert round(cvolT2, 1) == round(e.cvol.det.value, 1), "Cvol(T2) does not match expected value"
            # check error calculation
            eaux = round(e.cvol.det.value - e.cvol.t2, 1)
            assert eaux == round(e.cvol.det.error, 1), f"Error CVOL(T2) real {eaux} vs expected {e.cvol.det.error}"
            assert round(e.cvol.det.error_per, 2) == round(100 * abs(e.cvol.det.error / e.cvol.t2), 2), "Error percentage does not match expected value"
            assert round(e.evol.det.value, 1) == round(e.cvol.det.value - e.cvol.t1, 1), "Evol(T2) does not match expected value"


            # transform into km3 before printing
            q = bf.Qday_to_Qy(e.qperiod)
            cvol_real, cvol_det = bf.m3_to_km3(e.cvol.t2), bf.m3_to_km3(e.cvol.det.value)
            error, error_per = bf.m3_to_km3(e.cvol.det.error), e.cvol.det.error_per
            evol_real, evol_det = bf.m3_to_km3(e.evol.t2), bf.m3_to_km3(e.evol.det.value)

            print(f"{e.id}; {e.date.t2}; {dT:.0f} ; ", end=" ")
            print(f"{cvol_real:.6f}; {cvol_det:.6f};", end="")
            print(f"{error:.6f}; {error_per:.2f}%;", end="")
            print(f"{q:.4f} ;", end="\n")
            #print(f"{evol_real:.6f}; {evol_det:.6f}; {e.evol.det.error_per:.2f}%")

    @staticmethod
    def sanity_check_stoc(eruptions: list):
        """Print results of deterministic prediction."""
        print("Stochastic Forecast CVOL(T2) in km3:")

        print('ID ; DATE ; DT REAL (days) ; DT MEAN ; DT MEDIAN ; DT MODE ; '
              'CVOL REAL (km3) ; CVOL SIM MEAN ; ERROR ; % ; QHAT (km3/yr) ; '
              'CVOL REAL (km3); CVOL SIM MEDIAN ; ERROR ; % ; '
              'CVOL REAL (km3); CVOL SIM MODE ; ERROR ; % ;')

        for e in eruptions:

            if e.id == 1 or e.id == 74:
                continue  # skip first eruption and first of period II

            # parameters for deterministic method
            dT = e.dT.t1_2
            e_previous = eruptions[e.id - 2]
            assert e_previous.cvol.t2 == e.cvol.t1, f"Cumulative volume at T1 does not match previous eruption's CVOL(T2) - {e.id}: {e_previous.cvol.t2, e.cvol.t1}"

            # check state equation and error
            # cvolT2 = e.qperiod * dT + e.cvol.t1
            # assert round(cvolT2, 1) == round(e.cvol.det.value, 1), "Cvol(T2) does not match expected value"
            # # check error calculation
            # eaux = round(e.cvol.det.value - e.cvol.t2, 1)
            # assert eaux == round(e.cvol.det.error, 1), f"Error CVOL(T2) real {eaux} vs expected {e.cvol.det.error}"
            # assert round(e.cvol.det.error_per, 2) == round(100 * abs(e.cvol.det.error / e.cvol.t2), 2), "Error percentage does not match expected value"
            # assert round(e.evol.det.value, 1) == round(e.cvol.det.value - e.cvol.t1, 1), "Evol(T2) does not match expected value"
            #

            # transform into km3 before printing
            q = bf.Qday_to_Qy(e.qhat)
            cvol_real, cvol_sim_mean, cvol_sim_median, cvol_sim_mode = bf.m3_to_km3(e.cvol.t2), bf.m3_to_km3(e.cvol.sim.mean.value), bf.m3_to_km3(e.cvol.sim.median.value), bf.m3_to_km3(e.cvol.sim.mode.value)
            error_from_mean, error_per_from_mean = bf.m3_to_km3(e.cvol.sim.mean.error), e.cvol.sim.mean.error_per
            error_from_median, error_per_from_median = bf.m3_to_km3(e.cvol.sim.median.error), e.cvol.sim.median.error_per
            error_from_mode, error_per_from_mode = bf.m3_to_km3(e.cvol.sim.mode.error), e.cvol.sim.mode.error_per

            dT_mean, dT_median, dT_mode = e.dT.sim.mean.value, e.dT.sim.median.value, e.dT.sim.mode.value

            print(f"{e.id}; {e.date.t2}; {dT:.0f}; ", end=" ")
            # time intervals
            print(f"{dT_mean:.0f}; {dT_median:.0f} ; {dT_mode:.0f} ;", end=" ")
            # real vs mean
            print(f"{cvol_real:.6f}; {cvol_sim_mean:.6f};", end="")
            print(f"{error_from_mean:.6f}; {error_per_from_mean:.2f}%;", end="")
            print(f"{q:.4f}; ", end="")
            # median
            print(f"{cvol_real:.6f}; {cvol_sim_median:.6f};", end="")
            print(f"{error_from_median:.6f}; {error_per_from_median:.2f}%;", end="")
            # mode
            print(f"{cvol_real:.6f}; {cvol_sim_mode:.6f};", end="")
            print(f"{error_from_mode:.6f}; {error_per_from_mode:.2f}%;", end="\n")


    # ---------------------- OBSOLETE
