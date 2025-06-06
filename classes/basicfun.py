"""Functions I keep using all the time """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class basicfun:
    def __init__(self):
        self.name = "basicfun"

    # -----------------------------------------------------
    # BASIC STATS
    # -----------------------------------------------------

    @staticmethod
    def compute_mean_std(values: list) -> tuple:
        """Compute mean and standard deviation of a list of values"""
        if not values:
            return 0, 0
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        mean_value = np.mean(values)
        std_value = np.std(values)
        return mean_value, std_value

    @staticmethod
    def std_from_var(var: float) -> float:
        """Compute standard deviation from variance value"""
        if not var:
            return 0
        else:
            return np.sqrt(var)

    @staticmethod
    def get_limits(values: list) -> tuple:
        v_max = max(values)
        v_min = min(values)
        return v_min, v_max

    @staticmethod
    def get_total(values: list) -> float:
        return sum(values)

    @staticmethod
    def var_from_std(std: float) -> float:
        return std * std

    # VOLUME FUNCTIONS
    @staticmethod
    def compute_evol(cvol1: int, cvol2: int) -> int:
        """Compute Eruption Volume based on Cumulative Volume before/after eruption"""
        return cvol2 - cvol1

    @staticmethod
    def m3_to_km3(m3: float or list) -> float or list:
        """Convert m3 to km3"""
        if isinstance(m3, list):
            m3 = np.array(m3)
            m3 = m3 / 1e9
            km3 = list(m3)
        else:
            km3 = m3 / 1e9
        return km3

    # -----------------------------------------------------
    # TIME FUNCTIONS
    # -----------------------------------------------------
    @staticmethod
    def days_to_years(days: float) -> float:
        """Convert days to years"""
        years = days * 365.25
        return years

    @staticmethod
    def compute_intervals(dates: list):
        """Compute intervals between dates in days"""
        if not dates:
            return []
        intervals = []
        for i in range(1, len(dates)):
            delta_days = (dates[i] - dates[i - 1]).days
            intervals.append(delta_days)
        return intervals

    @staticmethod
    def compute_timeline(dT_days: list) -> list:
        """Compute the timeline of eruptions based on intervals"""
        timeline = [0]
        for dt in dT_days:
            previous_time = timeline[-1]
            timeline.append(previous_time + dt)
        return timeline

    # ------------------------------------------------------
    # RATE FUNCTIONS
    # ------------------------------------------------------
    @staticmethod
    def Qmday_to_kmy(Qd: float) -> float:
        """Convert Q units from m3/day to km3/year"""
        Qyears = (Qd / 1e9) * 365.25
        return Qyears

    @staticmethod
    def compute_q(cvol_tf: float, cvol_t0: float, dt_days: int) -> float:
        """Compute the rate Q in m3/day or km3/year"""
        delta_cvol = cvol_tf - cvol_t0
        if dt_days > 0:
            Q = delta_cvol / dt_days
        else:
            Q = 0
        return Q

    @staticmethod
    def get_q_line(q: float, dates: list, cvol: list) -> list:
        """Get a line of Q values for plotting"""
        n = len(dates)
        days = basicfun.compute_intervals(dates)

        cvol_theory = [cvol[0]]  # Start with the first cumulative volume
        for i in range(1, n):
            cvol_t2 = days[i - 1] * q + cvol[i]
            cvol_theory.append(cvol_t2)

        return cvol_theory

    # ------------------------------------------------------
    # PRINT FUNCTIONS
    # ------------------------------------------------------
    @staticmethod
    def print_period(t0, tf):
        """Print the period of measurements"""
        t1 = t0.strftime("%Y-%m-%d")
        t2 = tf.strftime("%Y-%m-%d")
        print(f"Period of measurements: {t1} -> {t2}")
        return t1, t2
