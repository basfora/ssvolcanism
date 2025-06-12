
"""Data class for a single eruption event."""

class OneEruption:
    """Class to represent a single eruption event with its date, volume, and cumulative volume."""

    def __init__(self):
        """
        Initialize the OneEruption instance.

        :param date: Date of the eruption.
        :param erupt_volume: Volume of the eruption.
        :param cum_volume: Cumulative volume up to this eruption.
        """

        self.date = EDate()

        self.evol = Evol()

        self.cvol = CVol()

        self.dT = TInterval()



class Evol:
    """Eruption volume"""

    def __init__(self):

        self.real = 0.0

        self.predicted = 0.0

        self.deterministic = 0.0

        self.linear = 0.0

        self.sim = Sim()


class CVol:
    """Cumulative volume"""

    def __init__(self):

        self.real = 0.0

        self.predicted = 0.0

        self.deterministic = 0.0

        self.linear = 0.0

        self.sim = Sim()


class EDate:
    """Eruption date"""

    def __init__(self):

        self.real = None

        self.predicted = None

        self.deterministic = None

        self.linear = None

        self.simulated = []

class TInterval:
    """Time interval between eruptions"""

    def __init__(self):

        self.real = 0.0

        self.predicted = 0.0

        self.deterministic = 0.0

        self.linear = 0.0

        self.sim = Sim()


class Sim:
    """Eruption simulation data"""

    def __init__(self):

        self.pts = []  # list of points (date, volume, cumulative volume)
        self.mean = 0.0
        self.std = 0.0
        self.median = 0.0


class EError:
    """Eruption error"""

    def __init__(self):

        self.abs = 0.0

        self.per = 0.0


