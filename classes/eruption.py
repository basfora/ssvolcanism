
"""Data class for a single eruption event."""

class OneEruption:
    """Class to represent a single eruption event with its date, volume, and cumulative volume."""

    def __init__(self):
        """
        Initialize the OneEruption instance.

        :param date: Date of the eruption.
        :param evol: Volume of the eruption.
        :param cvol: Cumulative volume up to this eruption.
        """

        self.date = EDate()

        self.evol = Vol()

        self.cvol = Vol()

        self.dT = TInterval()



class Vol:
    """Eruption or cumulative volume"""

    def __init__(self):

        self.real = 0.0
        self.real_error = EError()

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

        self.mean = 0.0
        self.std = 0.0

        self.lower = 0.0
        self.upper = 0.0

        self.mode = 0.0
        self.median = 0.0


class EDate:
    """Eruption date"""

    def __init__(self):

        self.real = None

        self.linear = None

        self.deterministic = None

        self.estimated = None

        self.sim = Sim()

class TInterval:
    """Time interval between eruptions"""

    def __init__(self):

        self.real = 0.0

        self.linear = 0.0

        self.deterministic = 0.0

        self.estimated = 0.0

        self.sim = Sim()


class EError:
    """Eruption error"""

    def __init__(self):

        self.abs = 0.0

        self.per = 0.0


