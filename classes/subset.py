import datetime

from classes.basicfun import Basicfun as bf


# for official periods, number 0+, subsets number -1
class MySubset:
    """Class to handle periods of interest for the volcano data
    :param period_number: integer representing the period number:
    -1 for subsets, 0 for all data, 1, 2, ... for official volcanism periods"""

    def __init__(self, period_number: int):
        """Initialize the period with eruption dates, volumes and cumulative volumes"""

        # identifier: -1 for custom subsets, 0 for all data, 1, 2, ... for official periods
        self.label = period_number

        # --------------------
        # Given by data file: Table<VolcanoName>.xlsx
        # --------------------
        # date of first and last eruption
        self.date_t0: datetime.date
        self.date_tf: datetime.date
        self.date_dT: datetime.timedelta  # time interval between t0 and tf

        # first and last eruption ID
        self.e0: int
        self.ef: int

        # rate for the period
        self.q_yr: float    # km3/year (used for visualization)
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

        # todo: number of eruptions and everything in compute for plotting

    def set_dates(self, datet0, datetf):
        """Set the start and end dates of the period"""
        if isinstance(datet0, datetime.date) and isinstance(datetf, datetime.date):
            self.date_t0 = datet0
            self.date_tf = datetf
            self.date_dT = bf.compute_days(datet0, datetf)  # time interval
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
            self.q_yr = q
            self.q = bf.Qy_to_Qday(q)
        else:
            self.q = q
            self.q_yr = bf.Qday_to_Qy(q)

    def set_cvol(self, cvol0, cvolf):
        """Set the cumulative volume at t0 and tf"""
        self.cvol_t0 = cvol0
        self.cvol_tf = cvolf

    def set_lists(self, edates, evols, cvols):
        """Set the lists of eruption dates, volumes and cumulative volumes"""
        if isinstance(edates, list) and isinstance(evols, list) and isinstance(cvols, list):
            self.edates = edates
            self.evols = evols
            self.cvols = cvols
        else:
            exit("Invalid list format. Use lists of datetime.date and float.")

        # compute time and stats after setting lists
        self.compute_time()
        self.compute_stats()

    # TODO NOW: compute for plotting stuff
    def compute_time(self):

        # intervals between eruptions
        self.intervals = bf.compute_intervals(self.edates)
        self.timeline = bf.compute_timeline(self.intervals, 0)


        return

    def compute_stats(self):

        # mean, median and mode for eruption volumes
        self.mean_evol, self.std_evol = bf.compute_mean_std(self.evols)
        self.median_evol = bf.compute_median(self.evols)




        return

    # TODO LATER: def sanity_check(number of eruption, q etc)