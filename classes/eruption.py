"""Data class for a single eruption event."""
import numpy as np
import datetime
from classes.basicfun import Basicfun as bf

class OneEruption:
    """Class to represent a single eruption event with its date, volume, and cumulative volume."""

    def __init__(self, eruption_id: int):
        """
        Initialize the OneEruption instance.
        :param eruption_id: # of eruption (1, 2, 3, ...)
        """

        # identifiers: eruption ID and which period it is in
        self.id = eruption_id
        self.period = 0
        # option to print results when True
        self.print = False

        # --------------- Variables at T2 (target variables)
        # date of the eruption (T2)
        self.date = EDate()
        # volume of the eruption EVOL(T2)
        self.evol = Vol()
        # cumulative volume of the eruption CVOL(T2)
        self.cvol = Vol()
        # time interval between eruptions dT = T2 - T1
        self.dT = TInterval()

        # ----------------- Parameters
        # qline parameters (a, b)
        self.a, self.b = None, None
        # deterministic: period rate of eruption computed from all eruptions in period (Q in m3/day)
        self.qperiod = None
        # stochastic: estimated rate of eruption until previous eruption (Qhat in m3/day)
        self.qhat = None


    def get_parameters(self, method=2):
        """Get parameters for each prediction methods."""

        # q-line
        if method == 1:
            cvolT1 = self.cvol.t0   # from beginning of period
            q = self.qperiod
            dT = self.dT.t0_2
        # deterministic
        elif method == 2:
            cvolT1 = self.cvol.t1
            q = self.qperiod
            dT = self.dT.t1_2 # real dT since last eruption
        # stochastic
        elif method == 3:
            cvolT1 = self.cvol.t1
            q = self.qhat
            dT = self.cvol.sim.N    # number of points for simulation, not real dT
        else:
            raise ValueError("Method must be 1, 2, 3 or 4.")

        return cvolT1, q, dT

    def save_real(self, edate: datetime.date, evol: int, cvol: int, when='t2'):
        """Save real (measured) data for the eruption"""

        if when == 't2':
            # save real data at T2
            self.date.t2 = edate
            self.evol.t2 = evol
            self.cvol.t2 = cvol
        elif when == 't1':
            # save real data at T1
            self.date.t1 = edate
            self.evol.t1 = evol
            self.cvol.t1 = cvol

        elif when == 't0':
            # save real data at T0
            self.date.t0 = edate
            self.evol.t0 = evol
            self.cvol.t0 = cvol

        else:
            raise ValueError("Invalid time point. Use 't0', 't1' or 't2'.")

        # start with checking if intervals are already computed
        if self.date.t2 is not None:
            # compute dT from T1 to T2 if T1 is available
            if self.date.t1 is not None:
                self.dT.t1_2 = bf.compute_days(self.date.t1, self.date.t2)
            # compute dT from T0 to T2 if T0 is available
            elif self.date.t0 is not None:
                self.dT.t0_2 = bf.compute_days(self.date.t0, self.date.t2)
            else:
                # do nothing, no previous date available
                pass

        return

    def save_parameter(self, param: tuple or float, method=1):

        if method == 1:  # qline
            self.a, self.b = param[0], param[1]
        elif method == 2:  # deterministic
            self.qperiod = param
        elif method == 3:  # stochastic
            self.qhat = param
        else:
            print(f"Method {method} invalid for saving parameters.")

    def save_result(self, dT, cvolT2, method=2):
        """Save the result of the prediction
        :param cvolT2: estimated cumulative vol (m3) at T2; float (methods 0-2), list (method 3)
        :param dT: estimated time interval (days); int (method 0-2), list (method 3)
        :param method: type of estimation:
        - method 0: real data
        - method 1: linear (qline)
        - method 2: deterministic
        - method 3: stochastic"""
        # todo: function can use a cleaning

        # error in CVOL
        cvol_error, cvol_error_per = bf.compute_error(self.cvol.t2, cvolT2)

        # compute EVOL at T2
        evolT2 = bf.compute_delta_vol(self.cvol.t1, cvolT2)
        evol_error, evol_error_per = bf.compute_error(self.evol.t2, evolT2)

        # compute date of the eruption
        date_ref = self.date.t1
        if date_ref is None:    # first eruption, no T1
            date_ref = self.date.t0
        if date_ref is None:  # no T0 either, use current date
            date_ref = self.date.t2
        dateT2 = bf.transform_days_to_date(dT, date_ref)

        if method == 0:  # real data
            self.cvol.t2 = cvolT2
            self.evol.t2 = evolT2
            self.dT.t2 = dT

        elif method == 1:   # q-linear
            self.cvol.qline.value = cvolT2
            self.cvol.qline.error, self.cvol.qline.error_per = cvol_error, cvol_error_per

            self.evol.qline.value = evolT2
            self.evol.qline.error, self.evol.qline.error_per = evol_error, evol_error_per

            self.dT.qline.value = dT
            self.dT.qline.error, self.dT.qline.error_per = 0, 0

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
                dt_er, dt_er_per = bf.compute_error(self.dT.t1_2, dt)
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
            n_bins = 20
            self.cvol.sim.mode.value = bf.compute_mode(cvolT2, n_bins)
            self.cvol.sim.mode.error, self.cvol.sim.mode.error_per = bf.compute_error(self.cvol.t2,
                                                                                      self.cvol.sim.mode.value)

            self.cvol.sim.lower, self.cvol.sim.upper = np.percentile(cvolT2, [2.5, 97.5])
            # evol
            self.evol.sim.mean.value, self.evol.sim.std = bf.compute_mean_std(evolT2)
            self.evol.sim.mean.error, self.evol.sim.mean.error_per = bf.compute_error(self.evol.t2,
                                                                                      self.evol.sim.mean.value)

            self.evol.sim.median.value = np.median(evolT2)
            self.evol.sim.median.error, self.evol.sim.median.error_per = bf.compute_error(self.evol.t2,
                                                                                          self.evol.sim.median.value)

            self.evol.sim.mode.value = bf.compute_mode(evolT2, n_bins)
            self.evol.sim.mode.error, self.evol.sim.mode.error_per = bf.compute_error(self.evol.t2,
                                                                                      self.evol.sim.mode.value)

            self.evol.sim.lower, self.evol.sim.upper = np.percentile(evolT2, [2.5, 97.5])

            # save mean and std dev for dT
            self.dT.sim.mean.value, self.dT.sim.std = bf.compute_mean_std(dT)
            self.dT.sim.mean.error, self.dT.sim.mean.error_per = bf.compute_error(self.dT.t1_2, self.dT.sim.mean.value)
            # save median and confidence interval for dT
            self.dT.sim.median.value = np.median(dT)
            self.dT.sim.median.error, self.dT.sim.median.error_per = bf.compute_error(self.dT.t1_2,
                                                                                      self.dT.sim.median.value)
            # save mode and error dT
            self.dT.sim.mode.value = bf.compute_mode(dT, n_bins)
            self.dT.sim.mode.error, self.dT.sim.mode.error_per = bf.compute_error(self.dT.t1_2,
                                                                                  self.dT.sim.mode.value)

            self.dT.sim.lower, self.dT.sim.upper = np.percentile(dT, [2.5, 97.5])
        else:
            raise ValueError("Method must be 1, 2, or 3.")

        return

    def print_instance(self, what=0):
        """Print the instance of the eruption data."""

        if what == 0: # real data
            if self.evol.t2 is None:
                print('No real data available.')
            else:
                first_line = 'REAL ERUPTION'
                evol = self.evol.t2
                cvol = self.cvol.t2
                edate = self.date.t2
                dT_days = self.dT.t1_2

                bf.print_one_eruption(self.id, evol, cvol, edate, dT_days)
            return

        elif what == 1:
            first_line = 'QLINE METHOD'
            evol = self.evol.qline.value
            cvol = self.cvol.qline.value

        elif what == 2:
            first_line = 'DETERMINISTIC METHOD'
            evol = self.evol.det.value
            cvol = self.cvol.det.value

        elif what == 3:
            first_line = 'STOCHASTIC METHOD - MEAN'
            evol = self.evol.sim.mean.value
            cvol = self.cvol.sim.mean.value
            bf.print_estimate(evol, cvol, first_line)

            first_line = 'STOCHASTIC METHOD - MEDIAN'
            evol = self.evol.sim.median.value
            cvol = self.cvol.sim.median.value
            bf.print_estimate(evol, cvol, first_line)

            first_line = 'STOCHASTIC METHOD - MODE'
            evol = self.evol.sim.mode.value
            cvol = self.cvol.sim.mode.value
        else:
            return

        # print
        bf.print_estimate(evol, cvol, first_line)


class Vol:
    """Eruption or cumulative volume"""

    def __init__(self):
        # real data (what really happened if available)
        self.t0 = None  # start of the period
        self.t1 = None  # right before eruption
        self.t2 = None  # at the end/after eruption
        # ------------ estimations

        # method I
        self.qline = EstimatedValue()
        # method II
        self.det = EstimatedValue()
        # method III
        self.stoc = EstimatedValue()
        self.sim = Sim()


class TInterval:
    """Time interval between eruptions (in days) """

    def __init__(self):
        # real data (what really happened if available)
        self.t0_2 = None # from beginning of the period to T2
        self.t1_2 = None # interval from T1 to T2
        # ------------ estimations

        # method I
        self.qline = EstimatedValue()
        # method II
        self.det = EstimatedValue()
        # method III
        self.stoc = EstimatedValue()
        self.sim = Sim()


class Sim:
    """Eruption simulation data"""

    def __init__(self):

        self.N = 10000
        self.pts = [EstimatedValue() for i in range(self.N)]  # list of points (date, volume, cumulative volume)

        self.mean = EstimatedValue()
        self.std = 0.0

        self.median = EstimatedValue()

        self.mode = EstimatedValue()

        self.lower = 0.0
        self.upper = 0.0


class EstimatedValue:

    def __init__(self):

        self.value = None
        self.error = None
        self.error_per = None

class EDate:
    """Eruption date (transform days in dates to plot if needed)"""

    def __init__(self):

        # todo maybe better way is to save edate and tinterval in one class,
        #  have a transform method to convert days to date

        # real data (what really happened if available)
        self.t0 = None
        self.t1 = None
        self.t2 = None
        # ------------ estimations

        # method I
        self.qline = EstimatedValue()
        # method II
        self.det = EstimatedValue()
        # method III
        self.stoc = EstimatedValue()
        self.sim = Sim()