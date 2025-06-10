"""Data used for analysis and prediction
One instance for each prediction pt"""
from classes.basicfun import basicfun as bf

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
        # CVOL(T2)
        self.cvol_hat = 0.0

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
        :param edates: list of TimeSeries [int]_n
        :param evol: list of eruption volumes (m3), [int]_n
        :param cvol: list of cumulative volume (m3) [int]_n
        :param id_last_eruption: number of last eruption"""

        if id_last_eruption is None:
            # if idx of last eruption is None, import all
            self.in_edates = edates
            self.in_evol = evol
            self.in_cvol = cvol
        else:
            # save only the data up to idx (recall: eruptions start at 1, python starts at 0)
            self.in_edates = edates[:id_last_eruption]
            self.in_evol = evol[:id_last_eruption]
            self.in_cvol = cvol[:id_last_eruption]

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

        # compute mean, std dev EVOL
        self.evol_mean, self.evol_std = bf.compute_mean_std(self.in_evol)

        # compute TIME
        self.dT_days = bf.compute_intervals(self.in_edates)
        self.dT_mean, self.dT_std = bf.compute_mean_std(self.dT_days)
        self.timeline = bf.compute_timeline(self.dT_days)
        self.dT_total = bf.compute_total_days(self.date_t0, self.date_tf)


        # sanity check (TODO move to UNIT test)
        assert sum(self.dT_days) == self.dT_total
        # TODO how to deal with CVOL in limits (e.g. cvol(T0) = 50 or 0 ?
        assert sum(self.in_evol) == self.cvol_delta + self.in_cvol[0]

        # compute RATE Q
        self.qhat = bf.compute_q(self.cvol_t0, self.cvol_tf, self.dT_total)

    def print_real_dataset(self):
        """Print info about the period of real data to be used for prediction"""

        bf.print_mark()
        bf.print_period_measurements(self.date_t0, self.date_tf)
        bf.print_n_eruptions(self.n)
        bf.print_vol_stats(self.evol_mean, self.evol_std, self.cvol_delta)
        bf.print_time(self.dT_mean, self.dT_std, self.dT_total)
        bf.print_rate(self.qhat)






    # compute intervals

    # prediction (stochastic forecast)

    # compare with real

    # output results and error

    # TO BE CALLED IN PLOTS: call this class IN PLOTS to plot stuff

    # todo COMPARE WITH REAL DATA >> save next eruption here or compare outside with collect data?






