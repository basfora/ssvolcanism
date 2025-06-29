
"""Data class for a single eruption event."""
from classes.basicfun import Basicfun as bf
import numpy as np
import datetime


class OneEruption:
    """Class to represent a single eruption event with its date, volume, and cumulative volume."""

    def __init__(self, eruption_id: int):
        """
        Initialize the OneEruption instance.

        :param eruption_id: # of eruption (1, 2, 3, ...)
        """

        self.id = eruption_id

        # --------------- variables at T2 (target variables)
        # date of the eruption (T2)
        self.date = EDate()
        # volume of the eruption EVOL(T2)
        self.evol = Vol()
        # cumulative volume of the eruption CVOL(T2)
        self.cvol = Vol()
        # time interval between eruptions dT = T2 - T1
        self.dT = TInterval()

        # -------------- variables at T1 (initial variables/parameters)
        # parameters specific for each prediction method (Q)
        self.qperiod = None
        # stochastic
        self.qhat = None
        # add more parameters if needed
        self.a, self.b = None, None  # for qline

    def get_parameters(self, method=2):
        """Get parameters for the prediction methods."""

        # same for all methods
        cvolT1 = self.cvol.t1

        if method == 1 or method == 4: # linear (regression)
            q = self.qperiod
            dT = self.dT.t2
        elif method == 2:   # deterministic
            # known (real) dT
            dT = self.dT.t2
            q = self.qperiod
        elif method == 3:   # stochastic
            dT = self.cvol.sim.N
            q = self.qhat
        elif method == 4:   # qline
            q = self.qperiod
            # from beginning of period
            dT = self.dT.t1
            cvolT1 = self.cvol.t0

        else:
            raise ValueError("Method must be 1, 2, or 3.")

        return cvolT1, q, dT

    def save_raw(self, edate: datetime.date, evol: int, cvol: int):

        self.date.t2 = edate
        self.cvol.t2 = cvol
        self.evol.t2 = evol

        return

    def save_result(self, cvolT2, dT, method=2):
        """Save the result of the prediction
        :param cvolT2: estimated cumulative vol (m3) at T2; float (methods 0-2), list (method 3)
        :param dT: estimated time interval (days); int (method 0-2), list (method 3)
        :param method: type of estimation:
        - method 0: real data
        - method 1: linear (change to qline)
        - method 2: deterministic
        - method 3: stochastic"""

        # error in CVOL
        cvol_error, cvol_error_per = bf.compute_error(self.cvol.t2, cvolT2)

        # compute evol
        evolT2 = bf.compute_delta_vol(self.cvol.t1, cvolT2)
        evol_error, evol_error_per = bf.compute_error(self.evol.t2, evolT2)

        # get estimated date
        dateT2 = bf.transform_days_to_date(dT, self.date.t1)

        if method == 0:  # real data
            self.cvol.t2 = cvolT2
            self.evol.t2 = evolT2
            self.dT.t2 = dT

        elif method == 1:   # linear
            self.cvol.linear.value = cvolT2
            self.cvol.linear.error, self.cvol.linear.error_per = cvol_error, cvol_error_per

            self.evol.linear.value = evolT2
            self.evol.linear.error, self.evol.linear.error_per = evol_error, evol_error_per

            self.dT.linear.value = dT
            self.dT.linear.error, self.dT.linear.error_per = 0, 0

            # save the date of the eruption
            self.date.linear = dateT2

        elif method == 2:  # deterministic
            self.cvol.det.value = cvolT2
            self.cvol.det.error, self.cvol.det.error_per = cvol_error, cvol_error_per

            self.evol.det.value = evolT2
            self.evol.det.error, self.evol.det.error_per = evol_error, evol_error_per

            self.dT.det.value = dT
            self.dT.det.error, self.dT.det.error_per = 0, 0  # no error for dT in deterministic method
            # save the date of the eruption
            self.date.deterministic = dateT2

        elif method == 3:  # stochastic -- inputs are lists of values CVOL, dT
            # save simulation results

            for i in range(len(cvolT2)):
                # CVOL
                self.cvol.sim.pts[i].value = cvolT2[i]
                self.cvol.sim.pts[i].error = cvol_error[i]
                self.cvol.sim.pts[i].error_per = cvol_error_per[i]
                # EVOL
                self.evol.sim.pts[i].value = evolT2[i]
                self.evol.sim.pts[i].error = evol_error[i]
                self.evol.sim.pts[i].error_per = evol_error_per[i]
                # DT
                dt = dT[i]
                dt_er, dt_er_per = bf.compute_error(self.dT.t2, dt)
                self.dT.sim.pts[i].value = dT[i]
                self.dT.sim.pts[i].error = dt_er
                self.dT.sim.pts[i].error_per = dt_er_per
                # date of the eruption
                self.date.sim.pts[i].value = dateT2[i]

            # statistics - mean, std dev, median, confidence interval
            self.cvol.sim.mean.value, self.cvol.sim.std = bf.compute_mean_std(cvolT2)
            self.cvol.sim.mean.error, self.cvol.sim.mean.error_per = bf.compute_error(self.cvol.t2,
                                                                                       self.cvol.sim.mean.value)
            self.cvol.sim.median.value = np.median(cvolT2)
            self.cvol.sim.median.error, self.cvol.sim.median.error_per = bf.compute_error(self.cvol.t2,
                                                                                        self.cvol.sim.median.value)

            self.cvol.sim.lower, self.cvol.sim.upper = np.percentile(cvolT2, [2.5, 97.5])
            # evol
            self.evol.sim.mean.value, self.evol.sim.std = bf.compute_mean_std(evolT2)
            self.evol.sim.mean.error, self.evol.sim.mean.error_per = bf.compute_error(self.evol.t2, self.evol.sim.mean.value)

            self.evol.sim.median.value = np.median(evolT2)
            self.evol.sim.median.error, self.evol.sim.median.error_per = bf.compute_error(self.evol.t2,
                                                                                      self.evol.sim.median.value)
            self.evol.sim.lower, self.evol.sim.upper = np.percentile(evolT2, [2.5, 97.5])
            # save mean and std dev for dT
            self.dT.sim.mean.value, self.dT.sim.std = bf.compute_mean_std(dT)
            self.dT.sim.mean.error, self.dT.sim.mean.error_per = bf.compute_error(self.dT.t2, self.dT.sim.mean.value)
            # save median and confidence interval for dT
            self.dT.sim.median.value = np.median(dT)
            self.dT.sim.median.error, self.dT.sim.median.error_per = bf.compute_error(self.dT.t2,
                                                                                      self.dT.sim.median.value)

            self.dT.sim.lower, self.dT.sim.upper = np.percentile(dT, [2.5, 97.5])

        elif method == 4: # qline
            self.cvol.qline.value = cvolT2
            self.cvol.qline.error, self.cvol.qline.error_per = cvol_error, cvol_error_per

            self.evol.qline.value = evolT2
            self.evol.qline.error, self.evol.qline.error_per = evol_error, evol_error_per

            self.dT.qline.value = dT
            self.dT.qline.error, self.dT.qline.error_per = 0, 0

            # save the date of the eruption
            dateT2 = bf.transform_days_to_date(dT, self.date.t0)
            self.date.linear = dateT2
        else:

            raise ValueError("Method must be 1, 2, or 3.")

        return

    def print_instance(self, what=0):

        if what == 0: # real data
            if self.evol.t2 is None:
                print('No real data available.')
            else:
                first_line = 'REAL ERUPTION'
                evol = self.evol.t2
                cvol = self.cvol.t2
                edate = self.date.t2
                dT_days = self.dT.t2
                # todo move to here, it's all over the place now
                bf.print_one_eruption(self.id, evol, cvol, edate, dT_days)
        elif what  ==1:
            evol = self.evol.linear.value
            cvol = self.cvol.linear.value
            bf.print_estimate(evol, cvol,'LINEAR EXTRAPOLATION')
        elif what == 2:
            evol = self.evol.det.value
            cvol = self.cvol.det.value
            bf.print_estimate(evol, cvol, 'DETERMINISTIC METHOD')
        elif what == 3:
            evol = self.evol.sim.mean.value
            cvol = self.cvol.sim.mean.value
            bf.print_estimate(evol, cvol, 'STOCHASTIC METHOD - MEAN')

            evol = self.evol.sim.mean.value
            cvol = self.cvol.sim.mean.value
            bf.print_estimate(evol, cvol, 'STOCHASTIC METHOD - MEDIAN')

        elif what == 4:
            evol = self.evol.det.value
            cvol = self.cvol.det.value
            bf.print_estimate(evol, cvol, 'QLINE METHOD')
        else:
            return



from datetime import date

class Vol:
    """Eruption or cumulative volume"""

    def __init__(self):
        # real data (what really happened if available)
        self.t0 = None
        self.t1 = None
        self.t2 = None
        # ------------ estimations

        # method I
        self.qline = EstimatedValue()
        self.linear = EstimatedValue()
        # method II
        self.det = EstimatedValue()
        # method III
        self.stoc = EstimatedValue()
        self.sim = Sim()


class TInterval:
    """Time interval between eruptions (in days) """

    def __init__(self):
        # real data (what really happened if available)
        self.t0 = None
        self.t1 = None
        self.t2 = None
        # ------------ estimations

        # method I
        self.qline = EstimatedValue()
        self.linear = EstimatedValue()
        # method II
        self.det = EstimatedValue()
        # method III
        self.stoc = EstimatedValue()
        self.sim = Sim()


class Sim:
    """Eruption simulation data"""

    def __init__(self):

        # todo transform in list of EstimatedValue (to include value, error, error_per)
        self.N = 10000
        self.pts = [EstimatedValue() for i in range(self.N)]  # list of points (date, volume, cumulative volume)

        self.mean = EstimatedValue()
        self.std = 0.0

        self.lower = 0.0
        self.upper = 0.0

        self.mode = 0.0
        self.median = EstimatedValue()


class EstimatedValue:

    def __init__(self):

        self.value = None
        self.error = None
        self.error_per = None

class EDate:
    """Eruption date (transform days in dates to plot if needed)"""

    def __init__(self):

        # todo maybe better way is to save edate and tinterval in one class, have a transform method to convert days to date

        # real data (what really happened if available)
        self.t0 = None
        self.t1 = None
        self.t2 = None
        # ------------ estimations

        # method I
        self.qline = EstimatedValue()
        self.linear = EstimatedValue()
        # method II
        self.det = EstimatedValue()
        # method III
        self.stoc = EstimatedValue()
        self.sim = Sim()