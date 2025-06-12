"""Data used for analysis and prediction
One instance for each prediction pt"""
from classes.basicfun import Basicfun as bf
from classes.eruption import OneEruption as oe
import numpy as np


class PredictionData:

    def __init__(self, edates: list, evol: list, cvol: list, idx=None, qtheory=None):

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

        # actual data for prediction
        self.real_date_t2 = None
        self.real_dT = None
        self.real_evol_t2 = None
        self.real_cvol_t2 = None
        self.qtheory = None

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

        # -----------------------------------------------------
        # METHOD 3: PREDICTED DATA
        # next eruption ID (for prediction)
        self.next_id = None
        # samples
        self.N = 1000
        # simulated time interval and cvol2
        self.sim_dT = None
        self.sim_cvolT2 = None

        # estimates (CHOSEN FROM SIM)
        self.cvolT2_hat = 0.0
        self.evolT2_hat = 0.0
        self.dT_hat = 0.0
        self.T2_hat = 0.0  # estimate date of next eruption

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
        # -----------------------------------------------------
        # METHOD 2: DETERMINISTIC
        self.cvolT2_det = None
        self.evolT2_det = None

        self.error_cvolT2_det = None
        self.error_evolT2_det = None
        self.error_evolT2_det_per = None
        self.error_cvolT2_det_per = None

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
        self.save_input_data(edates, evol, cvol, idx, qtheory)
        self.compute_real_stats()
        self.print_real_dataset()

    def save_input_data(self, edates: list, evol: list, cvol: list, id_last_eruption=None, qtheory=None):
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

        # save q theory if provided
        if qtheory is not None:
            self.qtheory = qtheory


        # number of data points for prediction
        self.n = len(self.in_evol)
        # next eruption ID
        self.next_id = self.n + 1

        # period
        self.date_t0 = self.in_edates[0]
        self.date_t1 = self.in_edates[-1]

        # cumulative volume (m3)
        self.cvol_t0 = self.in_cvol[0]
        self.cvol_t1 = self.in_cvol[-1]

    def compute_real_stats(self):

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

    def print_real_dataset(self):
        """Print info about the period of real data to be used for prediction"""

        bf.print_mark()
        bf.print_period(self.date_t0, self.date_t1)
        bf.print_n_eruptions(self.n)
        bf.print_vol_stats(self.evol_mean, self.evol_std, self.evol_sum)
        bf.print_cvol(self.cvol_t0, self.cvol_t1)
        bf.print_time(self.dT_mean, self.dT_std, self.time_total)
        bf.print_rate(self.qhat)

    def add_real_next(self, edate, evol: int, cvol: int):
        """Add real next eruption data (of what really happened) to compare with prediction
        :param dT: time interval (days) of next eruption
        :param evol: volume of next eruption (m3)
        :param cvol: cumulative volume at next eruption (m3)"""

        # save real data
        self.real_date_t2 = edate
        self.real_dT = bf.compute_days(self.in_edates[-1], self.real_date_t2)
        self.real_evol_t2 = evol
        self.real_cvol_t2 = cvol

    # ------------------------------------------------------------
    def run_prediction_methods(self):

        # stochastic forecast
        self.one_step_ahead()
        # choose estimate
        self.choose_estimate()
        self.compute_stats()
        # error
        self.forecast_error()

        # deterministic forecast
        self.deterministic()

        # PRINT
        bf.print_submark()
        # print real
        bf.print_one_eruption(self.next_id, self.real_evol_t2, self.real_cvol_t2,
                              self.real_date_t2, self.real_dT)

        # print results
        bf.print_deterministic(self.evolT2_det, self.cvolT2_det)

        bf.print_deterministic_error(self.error_evolT2_det, self.error_cvolT2_det,
                                     self.error_evolT2_det_per, self.error_cvolT2_det_per)


        # print results
        bf.print_prediction(self.evolT2_hat, self.cvolT2_hat, self.dT_hat,
                            (self.cvolT2_lower, self.cvolT2_upper))


        # print error
        bf.print_prediction_error(self.error_evolT2, self.error_cvolT2,
                                  self.error_evol_per, self.error_cvol_per, self.error_dT2)

        bf.print_mark()

    def organize_stuff(self):

        # TODO THIS AND PLOT!

        moe = oe()

        moe.evol.real = self.real_evol_t2
        print(f'ORGANIZE STUFF {moe.evol.real}')


    # ESTIMATION: NON-PARAMETRIC UNCERTAINTY PROPAGATION
    # (stochastic forecast)
    def one_step_ahead(self):
        """Predict NEXT eruption
        What I need for estimation:
        qhat, DT_list (days), T1 (timeline, day)
        and cvol(t1) = vcol[-1]"""

        # current cumulative volume at T1
        CV1 = [self.cvol_t1] * self.N
        dTdata = self.dT_days
        N = self.N
        qhat = self.qhat

        # ------------------------------------ SIMULATION
        # sampling time intervals
        # TODO change this for normal skewed?
        dTsim = np.random.choice(dTdata, N, replace=True)

        # compute CVOL(T2) = CVOL(T1) + Qhat * dTsim (for each dT)
        CV2 = CV1 + qhat * dTsim
        # ------------------------------------

        # save simulated data
        self.sim_cvolT2 = CV2
        self.sim_dT = dTsim

    def compute_stats(self):
        # STATS
        # compute mean and std of cv2_sim
        self.cvolT2_mean = np.mean(self.sim_cvolT2)
        self.cvolT2_std = np.std(self.sim_cvolT2)
        self.cvolT2_lower, self.cvolT2_upper = np.percentile(self.sim_cvolT2, [2.5, 97.5])

        # compute mean and std of dTsim
        self.dTsim_mean = np.mean(self.sim_dT)
        self.dTsim_std = np.std(self.sim_dT)
        self.dTsim_lower, self.dTsim_upper = np.percentile(self.sim_dT, [2.5, 97.5])


    def choose_estimate(self, best='median'):
        """Choose estimate, for now use mean or median of dTsim and cvolT2"""

        if best == 'mean':
            # choose mean
            CV2hat = np.mean(self.sim_cvolT2)
            dThat = np.mean(self.sim_dT)
        elif best == 'median':
            # choose median
            CV2hat = np.median(self.sim_cvolT2)
            dThat = np.median(self.sim_dT)
        else:
            # if no best is specified, use mean (for now)
            CV2hat = np.mean(self.sim_cvolT2)
            dThat = np.mean(self.sim_dT)

        # save estimates
        self.cvolT2_hat = CV2hat
        self.dT_hat = dThat  # next eruption time interval
        # next eruption volume
        self.evolT2_hat = self.cvolT2_hat - self.cvol_t1

    def forecast_error(self):
        """Compute forecast error between predicted and real data"""

        if self.real_dT is None or self.real_evol_t2 is None or self.real_cvol_t2 is None:
            print("Real next eruption data not set. Please use add_real_next() to set it.")
            return

        # compute ERROR
        self.error_dT2, _ = bf.compute_error(self.dT_hat, self.real_dT)
        self.error_evolT2, self.error_evol_per = bf.compute_error(self.evolT2_hat, self.real_evol_t2)
        self.error_cvolT2, self.error_cvol_per = bf.compute_error(self.cvolT2_hat, self.real_cvol_t2)

    def deterministic(self):
        """Compute cvol using method 1:
        deterministic with known T2, q from all period of real data
        """

        # compute cvol using method 2
        cvolT1 = self.cvol_t1
        dT = self.real_dT
        q = self.qtheory

        cvolT2 = bf.state_equation(cvolT1, dT, q)

        # save results
        self.cvolT2_det = cvolT2
        self.evolT2_det = cvolT2 - cvolT1

        # compute error
        self.error_cvolT2_det, self.error_cvolT2_det_per = bf.compute_error(cvolT2, self.real_cvol_t2)
        self.error_evolT2_det, self.error_evolT2_det_per = bf.compute_error(self.evolT2_det, self.real_evol_t2)

        return

    # TODO compare error between methods
    def error(self):
        """Compute error between
        prediction (method 3), deterministic with known T2 (method 2), q line (method 1)
        and real data"""


        # print error

        return


    # TO BE CALLED IN PLOTS: call this class IN PLOTS to plot stuff


