
"""Data class for a single eruption event."""
from classes.basicfun import Basicfun as bf


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
        self.q_line_xy = None
        self.q_linear = None
        self.q_period = None
        self.q_hat = None
        # add more parameters if needed

    def get_parameters(self, method=2):
        """Get parameters for the prediction methods."""

        # same for all methods
        cvolT1 = self.cvol.t1

        if method == 1:
            q = self.q_period
            dT = self.dT.real
        elif method == 2:   # deterministic
            # known (real) dT
            dT = self.dT.real
            q = self.q_period
        elif method == 3:   # stochastic
            # todo
            dT, q = None, None
        else:
            raise ValueError("Method must be 1, 2, or 3.")

        return cvolT1, q, dT

    def save_result(self, cvolT2, dT, method=2):
        """Save the result of the prediction."""

        # error in cvol
        cvol_error, cvol_error_per = bf.compute_error(cvolT2, self.cvol.real)

        # compute evol
        evolT2 = bf.compute_delta_vol(self.cvol.t1, cvolT2)
        evol_error, evol_error_per = bf.compute_error(evolT2, self.evol.real)
        dateT2 = bf.transform_days_to_date(dT, self.date.t1)

        if method == 0:  # q line
            # todo
            self.cvol.real = cvolT2

        elif method == 1:
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

        elif method == 3:  # stochastic
            self.cvol.stoc.value = cvolT2
            self.cvol.stoc.error, self.cvol.stoc.error_per = cvol_error, cvol_error_per

            self.evol.stoc.value = evolT2
            self.evol.stoc.error, self.evol.stoc.error_per = evol_error, evol_error_per

            self.dT.stoc.value = dT
            self.dT.stoc.error, self.dT.stoc.error_per = 0, 0
        else:
            raise ValueError("Method must be 1, 2, or 3.")

        return

    def print_instance(self, what=0):

        if what == 0: # real data
            if self.evol.real is None:
                print('No real data available.')
            else:
                first_line = 'REAL ERUPTION'
                evol = self.evol.real
                cvol = self.cvol.real
                edate = self.date.real
                dT_days = self.dT.real
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
        else:
            return



from datetime import date

class Vol:
    """Eruption or cumulative volume"""

    def __init__(self):

        # VOL(T1) data (known)
        self.t1 = None

        # real data (what really happened if avbailable)
        self.real = None

        self.qline = EstimatedValue()

        self.linear = EstimatedValue()

        self.det = EstimatedValue()

        # chosen estimate from simulation
        self.stoc = EstimatedValue()
        self.sim = Sim()


class TInterval:
    """Time interval between eruptions (in days) """

    def __init__(self):
        self.t1 = 0  # time of the first eruption in days
        self.real = None

        self.linear = EstimatedValue()

        self.qline = EstimatedValue()

        self.det = EstimatedValue()

        self.stoc = EstimatedValue()
        self.sim = Sim()


class Sim:
    """Eruption simulation data"""

    def __init__(self):

        self.pts = []  # list of points (date, volume, cumulative volume)
        self.N = 10000


        self.mean = 0.0
        self.std = 0.0

        self.lower = 0.0
        self.upper = 0.0

        self.mode = 0.0
        self.median = 0.0


class EstimatedValue:

    def __init__(self):

        self.value = None
        self.error = None
        self.error_per = None

class EDate:
    """Eruption date (transform days in dates to plot if needed)"""

    def __init__(self):

        # todo maybe better way is to save edate and tinterval in one class, have a transform method to convert days to date

        # T1 data (known)
        self.t1 = None

        self.real = None

        self.qline = None

        self.linear = None

        self.deterministic = None

        self.estimated = None
        self.sim = Sim()