import datetime

from classes.basicfun import Basicfun as bf


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
        # date of first and last eruption
        self.date_t0: datetime.date
        self.date_tf: datetime.date

        # first and last eruption ID
        self.e0: int
        self.ef: int

        # rate for the period
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
        # linear fit for cumulative volume
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

    def compute_qline(self):

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

