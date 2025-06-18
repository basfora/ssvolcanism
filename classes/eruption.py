
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
        self.q_linear = None
        self.q_period = None
        self.q_hat = None
        # add more parameters if needed

    def get_parameters(self, method=2):
        """Get parameters for the prediction methods."""

        # same for all methods
        cvolT1 = self.cvol.t1

        if method == 1:
            # todo
            dT, q = None, None
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
        # todo here: transform to date

        if method == 1:
            self.cvol.linear = cvolT2
            self.evol.linear = evolT2
        elif method == 2:  # deterministic
            self.cvol.deterministic = cvolT2
            self.cvol.deterministic_error = EError(cvol_error, cvol_error_per)

            self.evol.deterministic = evolT2
            self.evol.deterministic_error = EError(evol_error, evol_error_per)

            self.dT.deterministic = dT
            self.dT.deterministic_error = EError(0, 0)

        elif method == 3:  # stochastic
            self.cvol.estimated = cvolT2
            self.cvol.estimated_error = EError(cvol_error, cvol_error_per)

            self.evol.estimated = evolT2
            self.evol.estimated_error = EError(evol_error, evol_error_per)

            self.dT.estimated = dT
            self.dT.estimated_error = EError(0, 0)
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

        elif what == 2:
            evol = self.evol.deterministic
            cvol = self.cvol.deterministic
            bf.print_deterministic(evol, cvol)
        else:
            return



from datetime import date

class Vol:
    """Eruption or cumulative volume"""

    def __init__(self):

        # VOL(T1) data (known)
        self.t1 = 0

        # real data (what really happened if avbailable)
        self.real = None

        self.linear = 0.0
        self.linear_error: EError()

        self.deterministic = 0.0
        self.deterministic_error: EError()

        # chosen estimate from simulation
        self.estimated = 0.0
        self.estimated_error: EError()
        self.sim = Sim()


class TInterval:
    """Time interval between eruptions (in days) """

    def __init__(self):
        self.t1 = 0  # time of the first eruption in days
        self.real = None

        self.linear = 0.0
        self.linear_error = EError()

        self.deterministic = 0.0
        self.deterministic_error = EError()

        self.estimated = 0.0
        self.estimated_error = EError()
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


class EError:
    """Eruption error"""
    # TODO can be improved!

    def __init__(self, abs_error=0.0, per_error=0.0):

        self.abs = abs_error

        self.per = per_error


class EDate:
    """Eruption date (transform days in dates to plot if needed)"""

    def __init__(self):

        # todo maybe better way is to save edate and tinterval in one class, have a transform method to convert days to date

        # T1 data (known)
        self.t1 = None

        self.real = None

        self.linear = None

        self.deterministic = None

        self.estimated = None
        self.sim = Sim()