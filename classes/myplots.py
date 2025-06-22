"""Plots all in one place."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
import datetime

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

        # PLOTSET 1
        self.width = 14
        self.height = 6

        self.yth = 0.05  # y-axis limit for volume in m3
        self.short = 6

        # type data
        self.w_volume = "Volume"
        self.w_interval = "Interval"
        self.w_cvol = "Cumulative Volume"
        self.w_eruptions = "Eruptions"

        self.label_mean = "Mean = "
        self.label_mean_symbol = "$\mu$ = "
        self.label_median = "Median = "
        self.unit = [f"10$^{self.short}$ m$^3$",'days']

        # titles
        self.title_events = "Eruption Events"
        self.title_intervals = "Intervals between Eruptions"
        self.title_cvol = "Cumulative Volume"
        self.title_evol = "Eruption Volume"
        self.title_error_hist = "Error Distribution"
        self.title_error = "Error"
        self.title_exp = "Real vs Expected Data"
        self.title_linear = "Linear Extrapolation"
        self.title_qline = 'Q-line Approximation'

        # legends
        self.leg_real = "Real Data"
        self.leg_pred = "Predicted Data"
        self.leg_exp = "Expected"
        self.leg_Q1 = "Period I, Q = 0.0107 km$^3$/yr"
        self.leg_Q2 = "Period II, Q = 0.0228 km$^3$/yr"
        self.leg_error = "Error"

        # histogram
        self.n_bins = 20
        self.title_hist = "Distribution"

        # fig settings
        self.label_vol = f"Volume ($m^3$)"
        self.label_freq = "Frequency"
        self.label_date = "End of Eruption (Date)"
        self.label_interval =f"$\Delta~T (days)$"
        self.label_number = "Eruption Number"


        # COLORS, get from https://matplotlib.org/stable/gallery/color/named_colors.html
        self.color_real = 'navy'
        self.color_error = 'crimson' #'darkorange'
        self.color_mean = 'magenta' # 'darkgreen' # forestgreen'
        self.color_median = 'darkgreen' #'goldenrod'
        self.color_std = 'dimgray'
        self.color_hist = 'cornflowerblue'
        self.color_pred1 = 'brown' # red
        self.color_pred2 = 'salmon'# 'rosybrown' # magenta

        # linestyle/marker
        self.line_mean = '-'
        self.line_median = '-'
        self.line_std = '--'
        self.line_linear = '-'
        self.marker_real = 'x'
        self.marker_det = 'x'
        self.marker_error = '.'
        self.marker_linear = '.'

        self.savepath = self.get_save_path()

        if piton:
            self.set_piton()

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
        suptitle = f"{self.volcano_name} {self.w_eruptions} \n{self.period}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # -------------------- PLOT 1 (LEFT)
        if plot_op == 1:
            xvalues = vd.list_date
            yvalues = vd.list_eruptvol
            # mean and median for eruptions
            mean_value = vd.mean_evol
            median_value = vd.median_evol
            std_value = vd.std_evol
            label_std = f"Std Dev = {std_value/10**self.short:.2f} {self.unit[0]}"
            # label for volume
            label_mean = f"{mean_value/10**self.short:.2f} {self.unit[0]}"
            label_median = f"{median_value/10**self.short:.2f} {self.unit[0]}"
            labely = self.label_vol
            labelx = self.label_date
            # titles
            title1 = self.title_events
            title2 = f"Volume {self.title_hist}"
        else:
            xvalues = [i for i in range(1, len(vd.list_date))]
            yvalues = vd.intervals
            # mean and median for intervals
            mean_value = vd.mean_dT
            median_value = vd.median_dT
            std_value = vd.std_dT
            label_std = f"{std_value:.0f} {self.unit[1]}"
            # label for time
            label_mean = f"{mean_value:.0f} {self.unit[1]}"
            label_median = f"{median_value:.0f} {self.unit[1]}"
            labely = f"{self.label_interval}"
            labelx = self.label_number
            # titles
            title1 = self.title_intervals
            title2 = f"Time Interval {self.title_hist}"


        # PLOT ERUPTIONS (EVOL)
        ax[0].scatter(xvalues, yvalues, marker=self.marker_real, color=self.color_real, label=f"{self.w_eruptions}")
        # mean
        ax[0].axhline(mean_value, color=self.color_mean, linestyle=self.line_mean,
                   label=f"{self.label_mean} {label_mean}")
        # median
        ax[0].axhline(median_value, color=self.color_median, linestyle=self.line_mean,
                   label=f"{self.label_median} {label_median}")

        # std
        # ax[0].axhline(mean_value + std_value, color=self.color_std, linestyle=self.line_std,
        #               label=f"$\pm \sigma$ = {label_std}")
        # ax[0].axhline(mean_value - std_value, color=self.color_std, linestyle=self.line_std)

        # plot title, labels and legend
        ax[0].set(title=title1, xlabel=labelx, ylabel=labely)
        ax[0].legend(frameon=False)

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

        # std
        # ax[1].axvline(mean_value + std_value, color=self.color_std, linestyle=self.line_std,
        #               label=f"$\pm \sigma$ = {label_std}")
        # ax[1].axvline(mean_value - std_value, color=self.color_std, linestyle=self.line_std)

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

    def plot_real_vs_expected(self, eruptions: list, option='cvol', savename=None, show_plot=True):
        """Plot Volume (CVOL or EVOL) real and deterministic prediction"""

        self.period = bf.format_period(eruptions[0].date.real, eruptions[-1].date.real)

        suptitle = f"{self.volcano_name} - {self.period}"
        fig, ax = plt.subplots(1, 1, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # separate eruptions (class)
        ep1 = eruptions[:73]  # period I
        ep2 = eruptions[73:]  # period II
        self.sanity_piton_periods(ep1, ep2)

        # -------------------- DATA PREPARATION
        # xvalues: dates of eruptions
        xvalues = [e.date.real for e in eruptions]  # n
        xvalues1 = [e.date.real for e in ep1[1:]]  # Period I, skip first eruption, n-1
        xvalues2 = [e.date.real for e in ep2]  # Period II


        if option == 'cvol':
            myvol     = self.title_cvol
            # yvalues: real cumulative volumes (CVOL)
            yvalues_real = [e.cvol.real for e in eruptions]  # n
            # expected values (deterministic prediction)
            yvalues1 = [e.cvol.det.value for e in ep1 if e.cvol.det.value is not None]
            yvalues2 = [e.cvol.det.value for e in ep2 if e.cvol.det.value is not None]
        elif option == 'evol':
            myvol     = self.title_evol
            # yvalues: real eruption volumes (EVOL)
            yvalues_real = [e.evol.real for e in eruptions]  # n
            # expected values (deterministic prediction)
            yvalues1 = [e.evol.det.value for e in ep1 if e.evol.det.value is not None]
            yvalues2 = [e.evol.det.value for e in ep2 if e.evol.det.value is not None]
        else:
            exit("Option must be 'cvol' or 'evol'.")

        # ---------------- PLOT 1 (MAIN)
        # Plot real values (n)
        ax.scatter(xvalues, yvalues_real, marker=self.marker_real, color=self.color_real, linewidth=3, label=self.leg_real)
        # Plot predicted values - period I (eruptions # 2 - 73)
        ax.scatter(xvalues1, yvalues1, marker=self.marker_det, color=self.color_pred1, linewidth=1, label=self.leg_Q1)
        # Plot predicted values - period II (eruptions # 74 - end)
        ax.scatter(xvalues2, yvalues2, marker=self.marker_det, color=self.color_pred2, linewidth=1, label=self.leg_Q2)
        # plot title, labels and legend
        mytitle = f"{myvol}: {self.title_exp}"
        ax.set(title=mytitle, xlabel=self.label_date, ylabel=self.label_vol)
        ax.legend(frameon=False, loc='upper left')
        ax.grid(True)

        # set limits for x and y axes
        ylim = max(yvalues_real) * self.yth  # threshold for y-axis
        ax.set(ylim=(min(yvalues_real) - 2 * ylim, max(yvalues_real) + ylim))

        # show, save and close
        if show_plot:
            plt.show()
        if savename is None:
            savename = option
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

        self.period = bf.format_period(eruptions[0].date.real, eruptions[-1].date.real)

        suptitle = f"{self.volcano_name} {myvol} \n{self.period}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # ------------------------------------- PLOT 02: ERROR
        xvalues = [e.date.real for e in eruptions if e.id > 1]  # n-1

        # add stats: mean and std
        error_mean, error_std = bf.compute_mean_std(yvalues)

        # km3 for printing
        yvalues_km3 = bf.m3_to_km3(yvalues)
        # some error stats
        e_mean_km3, e_std_km3= bf.compute_mean_std(yvalues_km3)
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
        ax[0].set(title=f"{myvol}: {self.title_error}", xlabel=self.label_date, ylabel=self.label_vol)

        ax[0].legend(frameon=False, loc='upper left')
        ax[0].grid(True)

        # set limits for x and y axes
        ylim = max(yvalues) * self.yth
        ax[0].set(ylim=(min(yvalues) - 2 * ylim, max(yvalues) + ylim))

        # ------------------------------------- PLOT 2 (RIGHT) - ERROR HISTOGRAM
        # Plot histogram with KDE
        sns.histplot(yvalues, kde=True, ax=ax[1], color=self.color_hist, bins=self.n_bins)

        # plot identifiers
        plt.title(self.title_error_hist)
        # plot stats
        plt.axvline(error_mean, color=self.color_mean, linestyle=self.line_mean,
                    label=f"{self.label_mean_symbol} {e_mean_km3:.4f} km$^3$")
        plt.axvline(error_mean + error_std, color=self.color_std, linestyle=self.line_std,
                    label=f"$\pm \sigma$ = {e_std_km3:.4f} km$^3$")
        plt.axvline(error_mean - error_std, color=self.color_std, linestyle=self.line_std)

        plt.xlabel(self.label_vol)
        plt.ylabel(self.label_freq)
        plt.legend(frameon=False)

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

    def plot_linear(self, eruptions: list, option='cvol', method='linear', savename=None, show_plot=True):
        """Plot Volume (CVOL or EVOL) real and deterministic prediction"""

        self.period = bf.format_period(eruptions[0].date.real, eruptions[-1].date.real)

        suptitle = f"{self.volcano_name} - {self.period}"
        fig, ax = plt.subplots(1, 1, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # separate eruptions (class)
        ep1 = eruptions[:73]  # period I
        ep2 = eruptions[73:]  # period II

        # sanity check (date and q used)
        assert len(ep1) == 73, "Period I should have 73 eruptions"
        assert len(ep2) == 46, "Period II should have 46 eruptions"
        assert ep1[0].date.real == datetime.date(1936, 8, 1), "First date of period I should be 1936-1-8"
        assert ep1[-1].date.real == datetime.date(1998, 3, 11), "Last date of period I should be 1998-03-11"
        assert round(ep1[1].q_period, 4) == round(ep1[-1].q_period, 4) == round(bf.Qy_to_Qday(0.0107),
                                                                                4), "Q for period I should be 0.0107 km3/yr"

        assert ep2[0].date.real == datetime.date(1999, 7, 19), "First date of period II should be 1999-07-19"
        assert ep2[-1].date.real == datetime.date(2018, 7, 13), "First date of period II should be 2018-07-13"
        assert round(ep2[0].q_period, 4) == round(ep2[-1].q_period, 4) == round(bf.Qy_to_Qday(0.0228),
                                                                                4), "Q for period II should be 0.0228 km3/yr"

        # -------------------- DATA PREPARATION
        # xvalues: dates of eruptions
        xvalues = [e.date.real for e in eruptions]  # n
        xvalues1 = [e.date.real for e in ep1]  # Period I, skip first eruption, n-1
        xvalues2 = [e.date.real for e in ep2]  # Period II

        if option == 'cvol':
            myvol = f"{self.title_cvol} {self.title_linear}"
            # yvalues: real cumulative volumes (CVOL)
            yvalues_real = [e.cvol.real for e in eruptions]  # n
            # expected values (linear prediction)
            e1 = eruptions[0]  # first eruption, add so it can appear on plot
            yvalues1 = [e1.cvol.real] + [e.cvol.linear.value for e in ep1 if e.cvol.linear.value is not None]
            yvalues2 = [e.cvol.linear.value for e in ep2 if e.cvol.linear.value is not None]
        else:
            exit("Option must be 'cvol'")

        # ---------------- PLOT 1 (MAIN)
        # Plot real values (n)
        ax.scatter(xvalues, yvalues_real, marker=self.marker_real, color=self.color_real, linewidth=3,
                   label=self.leg_real)
        # Plot predicted values - period I (eruptions # 2 - 73)
        #ax.plot([xvalues1, xvalues2], [yvalues1, yvalues2], marker=self.marker_linear, color=self.color_pred1,)
        ax.plot(xvalues1, yvalues1, marker=self.marker_linear, color=self.color_pred1,
                   linewidth=1, label=self.leg_Q1)
        # Plot predicted values - period II (eruptions # 74 - end)
        ax.plot(xvalues2, yvalues2, marker=self.marker_linear, color=self.color_pred2, linestyle=self.line_linear,
                   linewidth=1, label=self.leg_Q2)

        # plot title, labels and legend
        mytitle = f"{myvol}: {self.title_exp}"
        ax.set(title=mytitle, xlabel=self.label_date, ylabel=self.label_vol)
        ax.legend(frameon=False, loc='upper left')
        ax.grid(True)

        # set limits for x and y axes
        ylim = max(yvalues_real) * self.yth  # threshold for y-axis
        ax.set(ylim=(min(yvalues_real) - 2 * ylim, max(yvalues_real) + ylim))

        # show, save and close
        if show_plot:
            plt.show()
        if savename is None:
            savename = option
        self.save_fig(fig, savename)

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
        assert len(ep1) == 73, "Period I should have 73 eruptions"
        assert len(ep2) == 46, "Period II should have 46 eruptions"
        assert ep1[0].date.real == datetime.date(1936, 8, 1), "First date of period I should be 1936-1-8"
        assert ep1[-1].date.real == datetime.date(1998, 3, 11), "Last date of period I should be 1998-03-11"
        assert round(ep1[1].q_period, 4) == round(ep1[-1].q_period, 4) == round(bf.Qy_to_Qday(0.0107),
                                                                                4), "Q for period I should be 0.0107 km3/yr"

        assert ep2[0].date.real == datetime.date(1999, 7, 19), "First date of period II should be 1999-07-19"
        assert ep2[-1].date.real == datetime.date(2018, 7, 13), "First date of period II should be 2018-07-13"
        assert round(ep2[0].q_period, 4) == round(ep2[-1].q_period, 4) == round(bf.Qy_to_Qday(0.0228),
                                                                                4), "Q for period II should be 0.0228 km3/yr"


    @staticmethod
    def sanity_check_det(eruptions: list):
        """Print results of deterministic prediction."""
        print("Deterministic Prediction CVOL(T2) in km3:")

        print('ID ; DATE ; CVOL REAL ; EXPECTED ; ERROR ; % ; Q (km3/day) ; DT (days) ; EVOL REAL ; EXPECTED')

        for e in eruptions[1:]:

            # parameters for deterministic method
            dT = e.dT.real
            e_previous = eruptions[e.id - 2]
            assert e_previous.cvol.real == e.cvol.t1, "Cumulative volume at T1 does not match previous eruption's CVOL(T2)"

            # check state equation and error
            cvolT2 = e.q_period * dT + e.cvol.t1
            assert round(cvolT2, 1) == round(e.cvol.det.value, 1), "Cvol(T2) does not match expected value"
            # check error calculation
            eaux = round(e.cvol.det.value - e.cvol.real, 1)
            assert eaux == round(e.cvol.det.error, 1), f"Error CVOL(T2) real {eaux} vs expected {e.cvol.det.error}"
            assert round(e.cvol.det.error_per, 2) == round(100* abs(e.cvol.det.error/e.cvol.real), 2), "Error percentage does not match expected value"
            assert round(e.evol.det.value, 1) == round(e.cvol.det.value - e.cvol.t1, 1), "Evol(T2) does not match expected value"


            # transform into km3 before printing
            q = bf.Qday_to_Qy(e.q_period)
            cvol_real, cvol_det = bf.m3_to_km3(e.cvol.real), bf.m3_to_km3(e.cvol.det.value)
            error, error_per = bf.m3_to_km3(e.cvol.det.error), e.cvol.det.error_per
            evol_real, evol_det = bf.m3_to_km3(e.evol.real), bf.m3_to_km3(e.evol.det.value)

            print(f"{e.id}; {e.date.real};", end=" ")
            print(f"{cvol_real:.6f}; {cvol_det:.6f};", end="")
            print(f"{error:.6f}; {error_per:.2f}%;", end="")
            print(f"{q:.4f}; {dT:.0f};", end="")
            print(f"{evol_real:.6f}; {evol_det:.6f}; {e.evol.det.error_per:.2f}%")

    # ---------------------- OBSOLETE
