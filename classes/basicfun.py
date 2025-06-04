"""Functions I keep using all the time """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class basicfun:
    def __init__(self):
        self.name = "basicfun"

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

    @staticmethod
    def Qmday_to_kmy(Qd: float) -> float:
        """Convert Q units from m3/day to km3/year"""
        Qyears = (Qd/1e9) *  365.25
        return Qyears

    @staticmethod
    def days_to_years(days: float) -> float:
        """Convert days to years"""
        years = days * 365.25
        return years

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
    def compute_mean_var(values: list) -> tuple:
        """Compute mean and standard deviation of a list of values"""
        if not values:
            return 0, 0
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        mean_value = np.mean(values)
        std_value = np.std(values)
        var_value = np.var(values)
        return mean_value, std_value, var_value


    @staticmethod
    def get_q_line(q: float, dates: list, cvol: list) -> list:
        """Get a line of Q values for plotting"""
        n = len(dates)
        days = basicfun.compute_intervals(dates)

        cvol_theory = [cvol[0]]  # Start with the first cumulative volume
        for i in range(1, n):
            cvol_t2 = days[i-1] * q + cvol[i]
            cvol_theory.append(cvol_t2)

        return cvol_theory

    @staticmethod
    def compute_intervals(dates: list):
        """Compute intervals between dates in days"""
        if not dates:
            return []
        intervals = []
        for i in range(1, len(dates)):
            delta_days = (dates[i] - dates[i-1]).days
            intervals.append(delta_days)
        return intervals

    @staticmethod
    def print_period(t0, tf):
        """Print the period of measurements"""
        t1 = t0.strftime("%Y-%m-%d")
        t2 = tf.strftime("%Y-%m-%d")
        print(f"Period of measurements: {t1} -> {t2}")
        return t1, t2

