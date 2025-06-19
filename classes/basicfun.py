"""Functions I keep using all the time """
import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class Basicfun:
    def __init__(self):
        self.name = "basicfun"

    # -----------------------------------------------------
    # BASIC STATS
    # -----------------------------------------------------

    # UT - ok
    @staticmethod
    def compute_mean_std(values: list) -> tuple:
        """Compute mean and standard deviation of a list of values
        Uses population standard deviation (ddof=0)
        To change that, use ddof=1 for sample standard deviation"""

        if not values:
            return 0, 0
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        mean_value = np.mean(values)
        # TODO check np.std with excel!!
        std_value = np.std(values, ddof=0)
        return mean_value, std_value


    @staticmethod
    def compute_median(values: list) -> float:
        """Compute median of a list of values"""
        if not values:
            return 0
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        median = np.median(values)
        return median


    # UT - ok
    @staticmethod
    def compute_error(valuehat: float, value: float) -> (float, float):
        """Compute error percentage"""
        if value == 0:
            return 0, 0
        else:
            error = (value - valuehat)
            error_per = abs(error/ value) * 100
            return error, error_per

    # UT - ok
    @staticmethod
    def std_from_var(var: float) -> float:
        """Compute standard deviation from variance value"""
        if not var:
            return 0
        else:
            return np.sqrt(var)

    # UT - ok
    @staticmethod
    def var_from_std(std: float) -> float:
        return std * std

    # UT - ok
    @staticmethod
    def get_limits(values: list) -> tuple:
        v_max = max(values)
        v_min = min(values)
        return v_min, v_max

    # UT - ok
    @staticmethod
    def get_total(values: list) -> float:
        return sum(values)

    # VOLUME FUNCTIONS
    # UT - ok
    @staticmethod
    def compute_delta_vol(cvol1: int or float, cvol2: int or float) -> int:
        """Compute Eruption Volume based on Cumulative Volume before/after eruption"""
        return cvol2 - cvol1

    # UT - ok
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
    # UT - ok
    @staticmethod
    def days_to_years(days: float) -> float:
        """Convert days to years"""
        years = days / 365.25
        return round(years, 4)

    # UT - ok
    @staticmethod
    def years_to_days(years: float) -> float:
        """Convert years to days"""
        days = years * 365.25
        return round(days, 4)

    # UT - OK
    @staticmethod
    def compute_intervals(dates: list) -> list:
        """Compute intervals between dates in days
        :return list of intervals in days (len(dates) - 1)"""
        if len(dates) < 2:
            return []

        intervals = []
        for i in range(1, len(dates)):
            delta_days = Basicfun.compute_days(dates[i], dates[i - 1])
            intervals.append(delta_days)

        return intervals

    # UT - OK
    @staticmethod
    def compute_days(date1, date2):
        """Compute the number of days between two dates"""
        return abs((date2 - date1).days)

    # UT - OK
    @staticmethod
    def compute_timeline(dT_days: list, first_day=0) -> list:
        """Compute the timeline of eruptions based on LIST OF INTERVALS"""
        timeline = [first_day]
        for dt in dT_days:
            previous_time = timeline[-1]
            timeline.append(previous_time + dt)
        return timeline

    # ------------------------------------------------------
    # RATE FUNCTIONS
    # ------------------------------------------------------
    # UT - ok
    @staticmethod
    def Qday_to_Qy(Qd: float) -> float:
        """Convert Q units from m3/day to km3/year"""
        Qyears = (Qd / 1e9) * 365.25
        return Qyears

    # UT - ok
    @staticmethod
    def Qy_to_Qday(Qy: float) -> float:
        """Convert Q units from m3/day to km3/year"""
        Qd = (Qy * 1e9) / 365.25
        return Qd

    @staticmethod
    def compute_q(cvol_t0: float, cvol_tf: float, dt_days: int) -> float:
        """Compute the rate Q in m3/day"""
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
        days = Basicfun.compute_intervals(dates)

        cvol_theory = [cvol[0]]  # Start with the first cumulative volume
        for i in range(1, n):
            cvol_t2 = days[i - 1] * q + cvol[i]
            cvol_theory.append(cvol_t2)

        return cvol_theory

    @staticmethod
    def state_equation(CV1: float, q: float, dT: float) -> float:
        """State equatiion"""

        # compute CVOL(T2) = CVOL(T1) + Qhat * dTsim (for each dT)
        CV2 = CV1 + q * dT

        return CV2

    # ------------------------------------------------------
    # PRINTING FUNCTIONS
    # ------------------------------------------------------
    @staticmethod
    def pevol(t: int):
        return 'Erupted Volume EVOL(t{})'.format(t)

    @staticmethod
    def pcvol(t: int):
        return 'Cumulative Volume CVOL(t{})'.format(t)


    @staticmethod
    def format_period(t1: datetime.date, t2: datetime.date) -> str:
        """Format period for printing"""

        link = "to"

        return f"{t1.strftime('%Y-%m-%d')} {link} {t2.strftime('%Y-%m-%d')}"

    @staticmethod
    def dec():
        return 1e6  # 1 million for m3 to km3 conversion

    @staticmethod
    def print_period(t0, tf):
        """Print the period of measurements"""
        t1 = t0.strftime("%Y-%m-%d")
        t2 = tf.strftime("%Y-%m-%d")
        print(f"Period of measurements: {t1} -> {t2}")
        return t1, t2

    @staticmethod
    def print_mark():
        print(f"=================================================")

    @staticmethod
    def print_submark():
        print(f"...................................................")

    @staticmethod
    def print_one_eruption(eid: int, evol, cvol, edate, dT_days):
        """Print one eruption (real)"""
        dec = Basicfun.dec()

        print(f"..........REAL ERUPTION")
        print(f"  Eruption ID: {eid}")
        print(f"  Date: {edate.strftime('%Y-%m-%d')} | dT = {dT_days:.0f} days")
        print(f"  {Basicfun.pevol(2)} = {evol / dec:.2f} ({dec} m3)")
        print(f"  {Basicfun.pcvol(2)} = {cvol / dec:.2f} ({dec} m3)")

    @staticmethod
    def print_prediction(evol_hat, cvol_hat, dT_hat, ci: tuple):

        dec = Basicfun.dec()

        print(f"..........STOCHASTIC METHOD")
        print(f"Mean time interval: {dT_hat:.0f} days after T1")
        print(f"{Basicfun.pevol(2)}: {evol_hat / dec:.4f} ({dec} m3)")
        print(f"{Basicfun.pcvol(2)}: {cvol_hat / dec:.4f} ({dec} m3)")
        print(f"95% CI: [{ci[0] / dec:.0f}, {ci[1] / dec:.0f}] ({dec} m3)")

    @staticmethod
    def print_deterministic(evol, cvol):
        """Print deterministic prediction"""
        dec = Basicfun.dec()

        print(f"..........DETERMINISTIC METHOD")
        print(f"{Basicfun.pevol(2)} = {evol / dec:.2f} ({dec} m3)")
        print(f"{Basicfun.pcvol(2)} = {cvol / dec:.2f} ({dec} m3)")


    @staticmethod
    def print_n_eruptions(n: int):
        print(f"Number of eruptions: {n}")

    @staticmethod
    def print_vol_stats(mean_value, std_value, delta_vol=None):

        # to visualize better
        dec = 1e6
        mean_value = mean_value / dec
        std_value = std_value / dec

        # Erupted Volume
        print(f"Volume ({dec:.0f} m3)")
        print(f"..Per eruption (EVOL): Mean: {mean_value:.4f} | Std: {std_value:.4f}")

        # cumulative volume
        if delta_vol is not None:
            print(f"..Total (SUM_EVOL): {delta_vol/dec:.2f}")

    @staticmethod
    def print_cvol(cvol1, cvol2):
        dec = 1e6

        print(f"..Initial, CVOL(t0) =  {cvol1/dec:.2f}")
        print(f"..Final, CVOL(t1) =  {cvol2/dec:.2f}")
        print(f"..Delta, dCVOL(t0->t1) = {(cvol2 - cvol1)/dec:.2f}")

    @staticmethod
    def print_time(mean_value, std_value, dt_total=None):

        # Erupted Volume
        print(f"Time interval between eruptions (days)")
        print(f"  Mean: {mean_value:.4f} | Std: {std_value:.4f}")

        # cumulative volume
        if dt_total is not None:
            print(f"  Total period: {dt_total}")

    @staticmethod
    def print_rate(q: float):
        """Rate q in mr/day"""

        qyears = Basicfun.Qday_to_Qy(q)
        print(f"Rate of eruptions: Q = {q:.4f} (m3/day) = {qyears:.4f} (km3/year)")

    @staticmethod
    def print_list(list_values: list, what: str):
        """For checking only"""
        print(f"{what} list (len = {len(list_values)}): {list_values}")


    @staticmethod
    def print_deterministic_error(eevol, ecvol, eevol_perc, ecvol_perc):
        """Print deterministic prediction error"""
        dec = 1e6

        print(f"..........DETERMINISTIC ERROR")
        print(f"Error {Basicfun.pevol(2)} = {eevol/dec:.2f} ({dec} m3) | {eevol_perc:.1f}%")
        print(f"Error {Basicfun.pcvol(2)} = {ecvol/dec:.2f} ({dec} m3) | {ecvol_perc:.1f}%")


    @staticmethod
    def print_prediction_error(eevol, ecvol, eevol_perc, ecvol_perc, edt):
        """Print prediction error"""
        dec = 1e6

        print(f"..........STHOCASTIC ERROR")
        print(f"  Time interval EdT = {edt:.0f} days")
        print(f"  Error {Basicfun.pevol(2)}  = {eevol/dec:.2f} ({dec:.0f} m3) | {eevol_perc:.1f}%")
        print(f"  Error {Basicfun.pcvol(2)}  = {ecvol/dec:.2f} ({dec:.0f} m3) | {ecvol_perc:.1f}%")
