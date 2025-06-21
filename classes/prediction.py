"""Data used for analysis and prediction
One instance for each prediction pt"""
from classes.basicfun import Basicfun as bf
from classes.eruption import OneEruption
import numpy as np


class PredictionData:

    def __init__(self, edates: list, evol: list, cvol: list, id_last=None):
        """Initialize the prediction data class
        :param edates: list of eruption dates [datetime](n)
        :param evol: list of eruption volumes (m3) [int](n)
        :param cvol: list of cumulative volumes (m3) [int](n+1)
        :param id_last: ID of last eruption (if None, use all data)
        """

        # for storage and information purposes
        # inputs (real data collection)
        self.in_edates = None
        self.in_evol = None
        self.in_cvol = None
        self.n = 0

        # period being used for prediction
        self.date_t0 = None
        self.date_t1 = None
        # current time, T1
        self.day_t1 = None

        # cumulative volume
        self.cvol_t0 = 0.0
        self.cvol_t1 = 0.0

        # eruption instance to save real and prediction data for target eruption
        self.oe = None  # id of next eruption

        # First things first, save input data
        self.save_input_data(edates, evol, cvol, id_last)

        # -----------------------------------------------------
        # COMPUTED (BASIC STATS) - to be used in analysis
        # EVOL (m3)
        self.evol_mean = 0.0
        self.evol_std = 0.0
        self.evol_sum = 0.0
        # CVOL (m3)
        self.cvol_delta = 0.0
        # TIME (DAYS)
        self.dT_days = []
        self.dT_mean, self.dT_std = 0.0, 0.0
        # for timeline
        self.time_total = 0.0
        self.timeline = []
        # RATE (m3/day)
        self.qhat = 0.0
        self.qperiod = None

        # -----------------------------------------------------
        # INIT FUNCTIONS
        self.comp_historical_stats()
        self.print_real_dataset()

    def save_input_data(self, edates: list, evol: list, cvol: list, id_last=None):
        """Just save input data directly and crate prediction instance
        :param edates: list of TimeSeries [int](n)
        :param evol: list of eruption volumes (m3), [int](n)
        :param cvol: list of cumulative volume (m3) [int](n+1)
        :param id_last: number ID of LAST known eruption"""

        # if id of last eruption is None, id = last available in dataset
        if id_last is None:
            id_last = len(evol) # number of eruptions (n) = last eruption id

        # save input data up to last eruption
        clip_here = id_last
        # actually save input data
        self.in_edates = edates[:clip_here]
        self.in_evol = evol[:clip_here]
        self.in_cvol = cvol[:clip_here + 1]  # adjust for fake 0 init
        # number of data points for prediction
        self.n = len(self.in_evol)

        # save T0 and T1 data for easy access
        # period
        self.date_t0 = self.in_edates[0]
        self.date_t1 = self.in_edates[-1]
        # time since beggining of period
        self.day_t1 = bf.compute_days(self.date_t0, self.date_t1)  # days from T0 to T1

        # cumulative volume (m3)
        self.cvol_t0 = self.in_cvol[0]
        self.cvol_t1 = self.in_cvol[-1]

        # ---------------------- CREATE ERUPTION INSTANCE
        # next eruption id
        self.next_id = id_last + 1  # next eruption ID (python starts at 0)
        self.oe = OneEruption(self.next_id)  # create an instance to save prediction data

        # save T1 data
        self.oe.date.t1 = self.date_t1
        self.oe.cvol.t1 = self.cvol_t1
        self.oe.dT.t1 = self.day_t1  # time interval in days from beginning of period to T1
        # not needed but for symmetry
        self.oe.evol.t1 = self.in_evol[-1]

        # save REAL T2 data if available
        if clip_here < len(evol):
            self.oe.date.real = edates[id_last]
            self.oe.evol.real = evol[id_last]
            self.oe.cvol.real = cvol[id_last + 1]  # cumulative volume at next eruption
            # interval
            self.oe.dT.real = bf.compute_days(self.date_t1, self.oe.date.real)

        bf.print_mark()
        print("Historical data saved, prediction instance created.")

    def comp_historical_stats(self):

        if self.in_evol is None:
            print(f"No data to analyse. Please restart program")
            exit()

        # compute delta volume CVOL
        self.cvol_delta = bf.compute_delta_vol(self.cvol_t0, self.cvol_t1)
        self.evol_sum = sum(self.in_evol)

        # compute mean, std dev EVOL
        self.evol_mean, self.evol_std = bf.compute_mean_std(self.in_evol)

        # compute TIME
        self.dT_days = bf.compute_intervals(self.in_edates)
        self.dT_mean, self.dT_std = bf.compute_mean_std(self.dT_days)
        self.timeline = bf.compute_timeline(self.dT_days)
        self.time_total = bf.compute_days(self.date_t0, self.date_t1)

        # compute RATE Q
        self.qhat = bf.compute_q(self.cvol_t0, self.cvol_t1, self.time_total)
        self.oe.q_hat = self.qhat  # save in OneEruption instance

    def print_real_dataset(self):
        """Print info about the period of real data to be used for prediction"""

        print('...')
        bf.print_period(self.date_t0, self.date_t1)
        bf.print_n_eruptions(self.n)
        bf.print_vol_stats(self.evol_mean, self.evol_std, self.evol_sum)
        bf.print_cvol(self.cvol_t0, self.cvol_t1)
        bf.print_time(self.dT_mean, self.dT_std, self.time_total)
        bf.print_rate(self.qhat)

    # ------------------------------------------------------------
    def set_qperiod(self, qperiod: float):
        """Set the theoretical rate of eruptions (m3/day)"""
        # save qperiod in OneEruption instance
        self.oe.q_period = qperiod

    def run_methods(self):

        # print real if available
        self.oe.print_instance(0)

        # deterministic method (2)
        self.deterministic_method()
        self.oe.print_instance(2)

        # linear extrapolation (1)
        # self.linear_extrapolation()
        # self.oe.print_instance(1)

        return self.oe


    # ------------------------------------------------------------
    # METHOD 2: DETERMINISTIC
    def deterministic_method(self):
        """Set the theoretical rate of eruptions (m3/day) for deterministic method"""

        # get parameters for deterministic method
        cvolT1, q, dT = self.oe.get_parameters(method=2)

        if q is None:
            print("No theoretical rate of eruptions (q) set. Please use set_qperiod() to set it.")
            return

        # cumulative volume at T2
        cvolT2 = bf.state_equation(cvolT1, q, dT)

        # save
        self.oe.save_result(cvolT2, dT, method=2)

    def stochastic_method(self):
        """Set the theoretical rate of eruptions (m3/day) for stochastic method"""

        # get parameters for deterministic method
        cvolT1, q, N = self.oe.get_parameters(method=2)

        # set up simulation parameters - time interval
        dTdata = self.dT_days  # use dT from historical data
        dTsim = np.random.choice(dTdata, N, replace=True)

        # compute CVOL(T2) = CVOL(T1) + Qhat * dTsim (for each dT)
        CV2 = [bf.state_equation(cvolT1, self.qhat, dT) for dT in dTsim]


        # save simulation results
        self.oe.save_result(CV2, dTsim, method=3)

    # -------------------------------------------------------------
    # METHOD 2: Linear extrapolation -- does not work like this (moved to VolcanoData class)
    def linear_extrapolation(self):
        """Set the theoretical rate of eruptions (m3/day) for linear extrapolation method"""

        # use timeline to fit to line
        xvalues = self.timeline
        yvalues = self.in_cvol[1:]

        # linear squares fit
        a, b = np.polyfit(xvalues, yvalues, 1)

        # use it to predict the next cumulative volume
        cvolT1, q, dT = self.oe.get_parameters(method=2)
        x2 = self.day_t1 + dT  # next time in timeline

        # extrapolate the next cumulative volume
        cvolT2 = a * x2+ b

        # save
        self.oe.save_result(cvolT2, dT, method=1)
        # save (test)
        self.oe.q_linear = (a, b)



