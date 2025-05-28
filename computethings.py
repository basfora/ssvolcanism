"""Computations for volcano"""
import datetime


class ComputeThings:

    def __init__(self, printing=True):
        """Initialize after collecting data"""

        # number of eruptions
        self.n = None
        # total time of measurements
        self.t0 = datetime.datetime
        self.tf = datetime.datetime
        self.dT_total = datetime.timedelta
        self.dT_days = 0
        self.dT_years = 0
        # eruptions / time
        self.avg_erupt_days = 0
        # interval between eruptions
        self.avg_dt_days = 0

        # volume
        # eruptions volume
        self.list_eruptvol = []
        # cum volume for checking
        self.list_cumvol = []
        # dates of eruptions
        self.list_dates = []
        # total cumulative volume (for rate calculations)
        self.cumvol_t0 = 0
        self.cumvol_tf = 0
        self.delta_cumvol = 0
        # list of time intervals between eruptions
        self.list_dt = []

        # rate
        self.Qd = 0
        self.Qy = 0

        # printing
        self.printing = printing


    # SET FUNCTIONS
    def set_period(self, t0, tf):
        """Set the period of measurements"""
        self.t0 = t0
        self.tf = tf
        self.dT_total = self.tf - self.t0

        self.dT_days = self.dT_total.days
        self.dT_years = self.tf.year - self.t0.year

    def set_n(self, n: int):
        """Set the number of eruptions"""
        self.n = n

    def compute_avg_eruption(self):
        """Set the average time between eruptions"""
        if self.n > 0:
            self.avg_dt_days = self.dT_days / self.n
        else:
            self.avg_dt_days = 0

    def print_simple_stats(self):
        """Print simple stats"""
        print(f"Number of eruptions: {self.n}")
        t1 = self.t0.strftime("%Y-%m-%d")
        t2 = self.tf.strftime("%Y-%m-%d")
        print(f"Period of measurements: {t1} -> {t2}")
        print(f"Time Interval: dT = {self.dT_days} days ({self.dT_years} years)")
        print(f"Average time between eruptions: {self.avg_dt_days:.2f} days")
        # -- volumes
        print(f".Total cumulative volume at t0 [{t1}]: {self.cumvol_t0:.0f} m3")
        print(f".Total cumulative volume at tf [{t2}]: {self.cumvol_tf:.0f} m3")
        print(f".Delta cumulative volume: {self.delta_cumvol:.0f} m3")
        # -- rates
        print(f"Rate of eruptions: {self.Qd:.4f} m3/day")
        print(f"Rate of eruptions: {self.Qy:.5f} km3/year")

    def set_data(self, dates, eruptvol, cumvol):
        """Set the data for analysis"""
        self.list_dates = dates
        self.list_eruptvol = eruptvol
        self.list_cumvol = cumvol

        # didn't set up n and time limits yet
        if self.n is None:
            n = len(self.list_dates)
            t0, tf = self.list_dates[0], self.list_dates[-1]
            self.set_n(n)
            self.set_period(t0, tf)
            self.compute_avg_eruption()

        # set cumulative volume at t0 and tf
        self.cumvol_t0 = self.list_cumvol[0]
        self.cumvol_tf = self.list_cumvol[-1]
        self.delta_cumvol = self.cumvol_tf - self.cumvol_t0

        # set rate of eruptions (based on data)
        self.compute_rate()

        if self.printing:
            self.print_simple_stats()


    def compute_rate(self):
        """Compute rate of volcano eruptions
        given cumulative volume and time"""

        # in days
        self.Qd = (self.cumvol_tf - self.cumvol_t0) / self.dT_days
        # in years
        self.Qy = self.Qdays_to_years(self.Qd)


    @staticmethod
    def Qdays_to_years(Qd: float):
        """Convert Qdays to Qyears"""
        Qyears = (Qd/1e9) *  365.25
        return Qyears



