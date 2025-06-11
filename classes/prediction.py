"""Data used for analysis and prediction
One instance for each prediction pt"""
from classes.basicfun import Basicfun as bf
import numpy as np


class PredictionData:

    def __init__(self, edates: list, evol: list, cvol: list, idx=None):

        # for storage and information purposes
        # inputs (real data collection)
        self.in_edates = None
        self.in_evol = None
        self.in_cvol = None
        self.n = 0

        # period being used for prediction
        self.date_t0 = None
        self.date_tf = None

        # cumulative volume
        self.cvol_t0 = 0.0
        self.cvol_tf = 0.0

        # actual data for prediction
        self.real_next_date = None
        self.real_next_dT = None
        self.real_next_evol = None
        self.real_next_cvol = None

        # COMPUTED (BASIC STATS) - to be used in analysis
        # EVOL (m3)
        self.evol_mean = 0.0
        self.evol_std = 0.0
        self.evol_sum = 0.0

        # CVOL (m3)
        self.cvol_delta = 0.0

        # TIME (DAYS)
        self.dT_days = []
        self.timeline = []
        self.dT_total = 0.0
        self.dT_mean, self.dT_std = 0.0, 0.0

        # RATE (m3/day)
        self.qhat = 0.0

        # -----------------------------------------------------
        # PREDICTED DATA
        # next eruption ID (for prediction)
        self.next_id = None
        # samples
        self.N = 1000
        # "current" time, T1
        self.t1 = None
        self.cvolT1 = None
        # simulated time interval and cvol2
        self.dTsim = None
        self.cvolT2 = None

        # STATS
        # first moment (time)
        self.dTsim_mean = 0.0
        self.dTsim_std = 0.0
        # first moment (cvol)
        self.cvolT2_mean = 0.0
        self.cvolT2_std = 0.0
        # confidence interval
        self.dTsim_lower, self.dTsim_upper = 0.0, 0.0
        self.cvolT2_lower, self.cvolT2_upper = 0.0, 0.0
        # estimates
        self.cvolT2_hat = 0.0
        self.evolT2_hat = 0.0
        self.dT_hat = 0.0
        self.T2_hat = 0.0 # estimate date of next eruption
        # -----------------------------------------------------

        # ERROR
        self.error_dT2 = None
        self.error_evolT2 = None
        self.error_cvolT2 = None
        # percentage error
        self.error_evol_per = None


        # TODO evol_all or only estimated one? (don't need to plot all)

        # -----------------------------------------------------
        # INIT FUNCTIONS
        self.save_input_data(edates, evol, cvol, idx)
        self.compute_real_stats()
        self.print_real_dataset()

    def save_input_data(self, edates: list, evol: list, cvol: list, id_last_eruption=None):
        """Save input real data for prediction, if idx is none, save all
        :param edates: list of TimeSeries [int](n)
        :param evol: list of eruption volumes (m3), [int](n)
        :param cvol: list of cumulative volume (m3) [int](n+1)
        :param id_last_eruption: number of last eruption"""

        if id_last_eruption is None:
            # if idx of last eruption is None, import all
            self.in_edates = edates
            self.in_evol = evol
            self.in_cvol = cvol # 0 init already in
        else:
            # save only the data up to idx (recall: eruptions start at 1, python starts at 0)
            self.in_edates = edates[:id_last_eruption]
            self.in_evol = evol[:id_last_eruption]
            self.in_cvol = cvol[:id_last_eruption + 1] # adjust for fake 0 init

            # we have real next data to compare later
            id_next = id_last_eruption # python index starts on zero
            edatenext, evolnext, cvolnext = edates[id_next], evol[id_next], cvol[id_next+1]
            self.add_real_next(edatenext, evolnext, cvolnext)

        # number of data points for prediction
        self.n = len(self.in_evol)
        # next eruption ID
        self.next_id = self.n + 1

        # period
        self.date_t0 = self.in_edates[0]
        self.date_tf = self.in_edates[-1]

        # cumulative volume (m3)
        self.cvol_t0 = self.in_cvol[0]
        self.cvol_tf = self.in_cvol[-1]

    def compute_real_stats(self):

        if self.in_evol is None:
            print(f"No data to analyse. Please restart program")
            exit()

        # compute delta volume CVOL
        self.cvol_delta = bf.compute_delta_vol(self.cvol_t0, self.cvol_tf)
        self.evol_sum = sum(self.in_evol)

        # compute mean, std dev EVOL
        self.evol_mean, self.evol_std = bf.compute_mean_std(self.in_evol)

        # compute TIME
        self.dT_days = bf.compute_intervals(self.in_edates)
        self.dT_mean, self.dT_std = bf.compute_mean_std(self.dT_days)
        self.timeline = bf.compute_timeline(self.dT_days)
        self.dT_total = bf.compute_days(self.date_t0, self.date_tf)

        # compute RATE Q
        self.qhat = bf.compute_q(self.cvol_t0, self.cvol_tf, self.dT_total)

    def print_real_dataset(self):
        """Print info about the period of real data to be used for prediction"""

        bf.print_mark()
        bf.print_period(self.date_t0, self.date_tf)
        bf.print_n_eruptions(self.n)
        bf.print_vol_stats(self.evol_mean, self.evol_std, self.evol_sum)
        bf.print_cvol(self.cvol_t0, self.cvol_tf)
        bf.print_time(self.dT_mean, self.dT_std, self.dT_total)
        bf.print_rate(self.qhat)

    # ------------------------------------------------------------
    # ESTIMATION: NON-PARAMETRIC UNCERTAINTY PROPAGATION
    # (stochastic forecast)
    def one_step_ahead(self):
        """Predict NEXT eruption
        What I need for estimation:
        qhat, DT_list (days), T1 (timeline, day)
        and cvol(t1) = vcol[-1]"""

        # todo TEST WITH PREDICTION ERUPTION 11
        # current cumulative volume at T1
        CV1 = [self.cvol_tf] * self.N
        dTdata = self.dT_days
        N = self.N
        qhat = self.qhat

        # ------------------------------------ SIMULATION
        # sampling time intervals
        dTsim = np.random.choice(dTdata, N, replace=True)

        # compute CVOL(T2) = CVOL(T1) + Qhat * dTsim (for each dT)
        CV2 = CV1 + qhat * dTsim
        # ------------------------------------

        # save simulated data
        self.cvolT2 = CV2
        self.dTsim = dTsim

        # STATS
        # compute mean and std of cv2_sim
        self.cvolT2_mean = np.mean(self.cvolT2)
        self.cvolT2_std = np.std(self.cvolT2)
        self.cvolT2_lower, self.cvolT2_upper = np.percentile(self.cvolT2, [2.5, 97.5])

        # compute mean and std of dTsim
        self.dTsim_mean = np.mean(self.dTsim)
        self.dTsim_std = np.std(self.dTsim)
        self.dTsim_lower, self.dTsim_upper = np.percentile(self.dTsim, [2.5, 97.5])

        # choose estimate
        self.choose_estimate()

        # print results
        bf.print_prediction(self.next_id, self.dT_hat, self.evolT2_hat, self.cvolT2_hat,
                            (self.cvolT2_lower, self.cvolT2_upper))

    def choose_estimate(self):
        """Choose estimate, for now use mean of dTsim and cvolT2"""

        self.cvolT2_hat = self.cvolT2_mean
        self.evolT2_hat = self.cvolT2_hat - self.cvol_tf  # next eruption volume
        self.dT_hat = self.dTsim_mean  # next eruption time interval

    def add_real_next(self, edate, evol: int, cvol: int):
        """Add real next eruption data (of what really happened) to compare with prediction
        :param dT: time interval (days) of next eruption
        :param evol: volume of next eruption (m3)
        :param cvol: cumulative volume at next eruption (m3)"""

        # save real data
        self.real_next_date = edate
        self.real_next_dT = bf.compute_days(self.in_edates[-1], self.real_next_date)
        self.real_next_evol = evol
        self.real_next_cvol = cvol

    def forecast_error(self):
        """Compute forecast error between predicted and real data"""

        if self.real_next_dT is None or self.real_next_evol is None or self.real_next_cvol is None:
            print("Real next eruption data not set. Please use add_real_next() to set it.")
            return

        # compute ERROR
        self.error_dT2, _ = bf.compute_error(self.dT_hat, self.real_next_dT)
        self.error_evolT2, self.error_evol_per = bf.compute_error(self.evolT2_hat, self.real_next_evol)
        self.error_cvolT2, self.error_cvol_per = bf.compute_error(self.cvolT2_hat, self.real_next_cvol)

        # print real
        bf.print_one_eruption(self.real_next_date, self.real_next_evol, self.real_next_cvol, self.real_next_dT)

        # print error
        bf.print_prediction_error(self.error_dT2, self.error_evolT2, self.error_cvolT2,
                                  self.error_evol_per, self.error_cvol_per)

        bf.print_mark()

    # TODO: add other methods for prediction
    def other_methods(self):
        """Compute cvol for
        method 1: q_rate line
        method 2: deterministic with known T2
        """

        return

    # TODO compare error between methods
    def error(self):
        """Compute error between
        prediction (method 3), deterministic with known T2 (method 2), q line (method 1)
        and real data"""


        # print error

        return


    # TO BE CALLED IN PLOTS: call this class IN PLOTS to plot stuff


