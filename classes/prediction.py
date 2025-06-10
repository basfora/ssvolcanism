"""Data used for analysis and prediction
One instance for each prediction pt"""
from classes.basicfun import basicfun as bf
import numpy as np


class PredictionData:

    def __init__(self, edates: list, evol: list, cvol: list, idx=None):

        # for storage and information purposes
        # inputs (real data collection)
        self.in_edates = None
        self.in_evol = None
        self.in_cvol = None
        self.n = 0

        # actual data for prediction
        self.real_next_date = None
        self.real_next_evol = None
        self.real_next_cvol = None

        # period being used for prediction
        self.date_t0 = None
        self.date_tf = None

        # cumulative volume
        self.cvol_t0 = 0.0
        self.cvol_tf = 0.0

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

        # PREDICTED DATA
        # samples
        self.N = 10000
        # "current" time
        self.t1 = None
        self.cvol_t1 = None
        # simulated time interval and cvol2
        self.dT_sim = None
        self.cvol_t2_sim = None
        # STATS
        # first moment (time)
        self.dT_sim_mean = 0.0
        self.dT_sim_std = 0.0
        # first moment (cvol)
        self.cvol_sim_mean = 0.0
        self.cvol_sim_std = 0.0
        # confidence interval
        self.dT_sim_lower, self.dT_sim_upper = 0.0, 0.0
        self.cvol_sim_lower, self.cvol_sim_upper = 0.0, 0.0
        # estimates
        self.cvol_hat = 0.0
        self.evol_hat = 0.0
        self.dT_hat = 0.0

        # TODO evol_all or only estimated one? (don't need to plot all)

        # REAL NEXT ERUPTION (for comparison)
        # todo come back later
        self.cvol_next = 0.0

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

        # number of data points for prediction
        self.n = len(self.in_evol)

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
        self.dT_total = bf.compute_total_days(self.date_t0, self.date_tf)

        # TODO move to UNIT test (sanity check)
        assert sum(self.dT_days) == self.dT_total
        assert sum(self.in_evol) == self.cvol_delta

        # compute RATE Q
        self.qhat = bf.compute_q(self.cvol_t0, self.cvol_tf, self.dT_total)

    def print_real_dataset(self):
        """Print info about the period of real data to be used for prediction"""

        bf.print_mark()
        bf.print_period_measurements(self.date_t0, self.date_tf)
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
        # TODO DO THIS BEFORE LEAVING TODAY!!

        # simple>
        # TODO check if need the self.N
        # T1 = self.timeline[-1]
        CV1 = self.cvol_tf
        dT_sim = np.random.choice(self.dT_days, self.N, replace=True)
        # compute cv(t2) = cv(t1) + qhat * dT_sim for each dT
        CV2 = CV1 + self.qhat * dT_sim
        # ------------------------------------
        # TODO change variable names (too convoluted!!)
        # current time and cvol(t1)
        self.t1 = [self.timeline[-1]] * self.N
        self.cvol_t1 = [self.cvol_tf] * self.N
        # sampling time intervals
        self.dT_sim = np.random.choice(self.dT_days, self.N, replace=True)
        # compute cv(t2) = cv(t1) + qhat * dT_sim for each dT
        self.cvol_t2_sim = self.cvol_t1 + self.qhat * self.dT_sim
        # compute mean and std of cv2_sim
        self.cvol_sim_mean = np.mean(self.cvol_t2_sim)
        self.cvol_sim_std = np.std(self.cvol_t2_sim)
        self.cvol_sim_lower, self.cvol_sim_upper = np.percentile(self.cvol_t2_sim, [2.5, 97.5])

        # TODO bf.print_sim_results
        bf.print_prediction(np.mean(self.dT_sim), self.cvol_sim_mean, (self.cvol_sim_lower, self.cvol_sim_upper))


    # output results and error

    # TO BE CALLED IN PLOTS: call this class IN PLOTS to plot stuff

    # todo COMPARE WITH REAL DATA >> save next eruption here or compare outside with collect data?
    # compare with real

    def set_real_next(self):
        """Get data of what really happened to compare with prediction"""

        return

    def other_methods(self):
        """Compute cvol for
        method 1: q_rate line
        method 2: deterministic with known T2
        """

        return

    def error(self):
        """Compute error between
        prediction (method 3), deterministic with known T2 (method 2), q line (method 1)
        and real data"""


        # print error

        return



