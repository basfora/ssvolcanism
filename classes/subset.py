import datetime
import numpy as np
from classes.basicfun import Basicfun as bf
from classes.eruption import OneEruption


# for official periods, number 0+, subsets number -1
class MySubset:
    """Class to handle periods of interest for the volcano data
    :param period_number: integer representing the period number:
    -1 for subsets, 0 for all data, 1, 2, ... for official volcanism periods"""

    def __init__(self, period_number: int):
        """Initialize the period with eruption dates, volumes and cumulative volumes
        :param period_number: int, period identifier
        Use -1 for custom subsets, 0 for all data, 1, 2, ... for official periods"""

        # identifier
        self.label = period_number
        self.volcano_name: str

        # --------------------
        # Given by data file: Table<VolcanoName>.xlsx
        # --------------------
        # date of first and last eruption IN SUBSET
        self.date_t0: datetime.date
        self.date_tf: datetime.date

        # first and last eruption ID
        self.e0: int
        self.ef: int

        # rate for the subset
        self.qyr: float    # km3/year (used for visualization)
        self.q: float       # m3/day (used for calculations)

        # actual eruption data within edate[t0, tf]
        self.edates = list()    # list of eruption dates (datetime.date)
        self.evols = list()     # list of eruption volumes (m3)
        self.cvols = list()     # list of cumulative volumes (m3)

        # --------------------
        # Computed values
        # --------------------
        # number of eruptions
        self.n: int
        # Cumulative Volume (cvol)
        self.cvol_t0: float = 0.0  # cumulative volume at t0
        self.cvol_tf: float = 0.0  # cumulative volume at tf
        self.cvol_delta: float = 0.0  # delta volume (cvol_tf - cvol_t0)
        # eruption volume
        self.evol_sum: float = 0.0  # sum of eruption volumes
        # mean, median and mode for eruption volumes
        self.evol_mean: float = 0.0
        self.evol_std: float = 0.0
        self.evol_median: float = 0.0
        self.evol_mode: float = 0.0
        # intervals between eruptions
        self.intervals = list()  # list of intervals between eruptions (days)
        self.timeline = list()  # timeline of intervals (days)
        # mean, median and mode for eruption intervals
        self.dT_total_days: int = 0
        self.dT_mean: float = 0.0
        self.dT_std: float = 0.0
        self.dT_median: float = 0.0
        self.dT_mode: float = 0.0

        # --------------------
        # Prediction
        # --------------------
        self.next_id: int = 0  # next eruption ID to predict

        # METHOD 1: Q-line fit for cumulative volume (WHEN LABEL >= 0)
        self.a: float = 0.0  # slope
        self.b: float = 0.0  # intercept
        self.line_points = list()  # points for the linear fit line (x, y)

    def set_vname(self, vname: str):
        """Set the volcano name for the period"""
        if isinstance(vname, str):
            self.volcano_name = vname
        else:
            exit("Invalid volcano name format. Use a string.")

    def set_dates(self, datet0, datetf):
        """Set the start and end dates of the period"""
        if isinstance(datet0, datetime.date) and isinstance(datetf, datetime.date):
            self.date_t0 = datet0
            self.date_tf = datetf
            self.dT_total_days = bf.compute_days(datet0, datetf)  # time interval
        else:
            exit("Invalid date format. Use datetime.date objects.")

    def set_eIDs(self, eid_t0, eid_tf):
        """Set the first and last eruption IDs of the period"""
        if isinstance(eid_t0, int) and isinstance(eid_tf, int):
            self.e0 = eid_t0
            self.ef = eid_tf
            self.n = eid_tf - eid_t0 + 1  # number of eruptions
        else:
            exit("Invalid eruption ID format. Use integers.")

    def set_q(self, q: float, opt='yr'):
        """Set the rate for the period
        :param q: float, rate in m3/day or km3/year
        :param opt: 'yr' for km3/year, 'day' for m3/day"""

        if opt == 'yr':
            self.qyr = q
            self.q = bf.Qy_to_Qday(q)
        else:
            self.q = q
            self.qyr = bf.Qday_to_Qy(q)

    def set_cvol(self, cvol0, cvolf):
        """Set the cumulative volume at t0 and tf"""
        self.cvol_t0 = cvol0
        self.cvol_tf = cvolf
        self.cvol_delta = bf.compute_delta_vol(cvol0, cvolf)  # delta volume

    def set_lists(self, edates, evols, cvols):
        """Set the lists of eruption dates, volumes and cumulative volumes"""
        if isinstance(edates, list) and isinstance(evols, list) and isinstance(cvols, list):
            self.edates = edates
            self.evols = evols
            self.cvols = cvols
        else:
            exit("Invalid list format. Use lists of datetime.date and float.")

        # compute time and stats after setting lists
        self.compute_stats()

    def compute_stats(self):

        # intervals between eruptions
        self.intervals = bf.compute_intervals(self.edates)
        self.timeline = bf.compute_timeline(self.intervals, 0)

        # EVOL
        self.evol_sum = sum(self.evols)
        # mean, median and mode for eruption volumes
        self.evol_mean, self.evol_std = bf.compute_mean_std(self.evols)
        self.evol_median = bf.compute_median(self.evols)
        self.evol_mode = bf.compute_mode(self.evols)

        # mean, median and mode for eruption intervals
        self.dT_mean, self.dT_std = bf.compute_mean_std(self.intervals)
        self.dT_median = bf.compute_median(self.intervals)
        self.dT_mode = bf.compute_mode(self.intervals)

        # non-official period, needs to compute q based on data
        if self.label < 1:
            q = bf.compute_q(self.cvol_t0, self.cvol_tf, self.dT_total_days)
            self.set_q(q, 'day')

        return

    # ----------------------------------------------------------
    # PREDICTION METHODS
    # ----------------------------------------------------------
    def run_prediction(self, nextoe: OneEruption):
        """Run the prediction methods for the next eruption instance"""

        # set the next eruption ID
        self.next_id = nextoe.id

        # save subset-related info (might not be a full period)
        nextoe.save_parameter(self.q, 3)  # set theoretical rate of eruptions (m3/day)

        # T0: real data for first eruption in SUBSET
        nextoe.save_real(self.date_t0, self.evols[0], self.cvol_t0, 't0')

        # run methods for prediction
        oe = self.run_all_methods(nextoe)

        return oe

    def run_all_methods(self, oe: OneEruption):
        """Run all prediction methods for the given eruption instance"""

        # qline method (1)
        oe = self.qline_method(oe)

        # deterministic method (2)
        oe = self.deterministic_method(oe)

        # stochastic method (3)
        oe = self.stochastic_method(oe)

        print(f"Predicting eruption {oe.id} using eruptions {self.e0} - {self.ef}")
        if oe.print:
            print('--------------------------------------------------')
            # print real if available
            oe.print_instance(0)
            oe.print_instance(1)
            oe.print_instance(2)
            oe.print_instance(3)

        return oe

    # METHOD 1: Q-LINE FIT
    def qline_method(self, oe: OneEruption):

        # if the period is NOT official, prioritize eruption instance info
        if self.label < 0:
            # only run if the eruption instance has NO valid Q-line parameters
            if oe.a is None:
                print(f"Warning: Q-line parameters not set for eruption {oe.id}. "
                      f"Using subset parameters instead.")

                # Q-LINE (get or compute) for this eruption
                qpt = self.get_qline_pt(oe.id)
                # save results in the eruption instance
                oe.save_result(qpt[0], qpt[1], method=1)
                # save the slope and intercept for the Q-line fit in eruption instance
                oe.save_parameter((self.a, self.b), 1)

        # if the period IS official, prioritize subset (period) info
        elif self.label >= 0:
            if oe.a is not None:
                print(f"Warning: Q-line parameters already set for eruption {oe.id}. "
                      f"Overwriting them with period parameters.")

            # Q-LINE (get or compute) for this eruption
            qpt = self.get_qline_pt(oe.id)
            # save results in the eruption instance
            oe.save_result(qpt[0], qpt[1], method=1)
            # save the slope and intercept for the Q-line fit in eruption instance
            oe.save_parameter((self.a, self.b), 1)
        else:
            return None  # no valid period, cannot run Q-line method


        return oe

    def get_qline_pt(self, eid: int, opt='dT')-> tuple:
        """Get the points for the Q-line fit for a given eruption ID
        To be used when subset is an official period"""

        idx = eid - self.e0  # index of the eruption in the list

        # check if the index is valid
        if idx < 0 or idx >= self.n:
            raise ValueError(f"Eruption ID {eid} is out of bounds for the period {self.label}.")

        # check if the line points are computed
        if len(self.line_points) == 0:
            # raise warning if computing from given subset (not period)
            if self.label < 0:
                print(f"Warning: Not an official period. "
                      f"Computing Q-line points for subset of eruptions {self.e0} - {self.ef}.")
            self.compute_qline()

        # RETRIEVE Q-LINE POINT: (t2, cvol(t2))
        pt = self.line_points[idx]  # get the point for the Q-line fit

        # return the point as (t02, cvol(t2))
        if opt != 'dT':
            return pt

        # transform timeline into interval since previous eruption (to align with the other methods)
        t2 = pt[0]  # time at T2 in timeline
        t1 = self.timeline[idx-1] if idx > 0 else 0  # time at T1 in timeline
        dT1_2 = t2 - t1  # time interval from T1 to T2

        # cvol(t2) is the cumulative volume at T2
        cvol_t2 = pt[1]
        # create and return point as a tuple
        qpt = (dT1_2, cvol_t2)

        return qpt

    def compute_qline(self):
        """Compute the Q-line fit for cumulative volume over time"""

        xvalues = self.timeline
        yvalues = self.cvols[1:]

        x1, y1 = xvalues[0], yvalues[0]
        x2, y2 = xvalues[-1], yvalues[-1]
        a = (y2 - y1) / (x2 - x1)  # slope
        b = y1 - a * x1  # intercept

        # save the slope and intercept
        self.a, self.b = a, b

        # create points for the line based on fit and timeline
        self.line_points = [(x, a * x + b) for x in xvalues]

        print(f"... Q-line fit computed for period {self.label}")

    # METHOD 2: DETERMINISTIC
    @staticmethod
    def deterministic_method(oe: OneEruption):
        """Set the theoretical rate of eruptions (m3/day) for deterministic method"""

        # get parameters for deterministic method
        cvolT1, q, dT = oe.get_parameters(method=2)

        if q is None:
            q = oe.qhat
            print("Warning: No theoretical rate of eruptions set for the period. "
                  "Using computed rate qhat instead.")

        # cumulative volume at T2
        cvolT2 = bf.state_equation(cvolT1, q, dT)

        # save
        oe.save_result(dT, cvolT2, method=2)
        return oe

    # METHOD 3: STOCHASTIC FORECAST
    def stochastic_method(self, oe: OneEruption):
        """Set the theoretical rate of eruptions (m3/day) for stochastic method"""

        # get parameters for stochastic method
        cvolT1, qhat, N = oe.get_parameters(method=3)

        # set up simulation parameters - time interval
        dTdata = self.intervals  # use dT from historical data
        if len(dTdata) == 0:
            dTdata = np.random.randint(2, 1200, 20)  # if no data, use random dT

        dTsim = np.random.choice(dTdata, N, replace=True)

        # compute CVOL(T2) = CVOL(T1) + Qhat * dTsim (for each dT)
        CVsim = [bf.state_equation(cvolT1, qhat, dT) for dT in dTsim]

        # save simulation results
        oe.save_result(dTsim, CVsim, method=3)
        return oe

    def print_real_dataset(self):
        """Print info about the period of real data to be used for prediction"""

        print('...')
        bf.print_period(self.date_t0, self.date_tf)
        bf.print_n_eruptions(self.n)
        bf.print_vol_stats(self.evol_mean, self.evol_std, self.evol_sum)
        bf.print_cvol(self.cvol_t0, self.cvol_tf)
        bf.print_time(self.dT_mean, self.dT_std, self.dT_total_days)
        bf.print_rate(self.qyr)

    def sanity_check(self):
        # todo expand this as needed

        assert self.n == self.ef - self.e0 + 1, "Number of eruptions does not match IDs"
        assert len(self.edates) == self.n, "Number of eruption dates does not match n"
        assert len(self.evols) == self.n, "Number of eruption volumes does not match n"
        assert len(self.cvols) == self.n + 1, "Number of cumulative volumes does not match n"

        # time
        assert self.dT_total_days == sum(self.intervals), "Total days does not match intervals sum"
        assert len(self.intervals) == self.n - 1, "Number of intervals does not match n-1"
        assert len(self.timeline) == self.n, "Timeline length does not match n"
        assert self.timeline[-1] == self.dT_total_days, "Last timeline value does not match total days"

