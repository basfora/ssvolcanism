"""Data used for analysis and prediction
One instance for each prediction pt"""
from classes.basicfun import Basicfun as bf
from classes.eruption import OneEruption
import numpy as np

from classes.subset import MySubset


class PredictionData:

    def __init__(self, nextid: int):
        """Initialize the prediction data class
        """
        # input data for prediction
        self.indata = None
        self.n = 0  # number of data points

        # next eruption ID (to be used in prediction)
        self.next_id = nextid
        print(f"Prediction instance created for eruption id # {self.next_id}")
        self.oe = None

        self.a = dict()  # dictionary to save a, b for qline method
        self.b = dict()
        self.line_pts = dict()  # dictionary to save line points for qline method


    def input_dataset(self, subdata: MySubset):

        self.indata = subdata
        self.n = subdata.n  # number of data points

    def input_eruption(self, oe: OneEruption):

        assert oe.id == self.next_id, "Eruption ID does not match the next ID for prediction."

        # save t0 data in eruption instance
        edate, evol, cvol = self.indata.date_t0, self.indata.evols[0], self.indata.cvol_t0
        oe.save_real(edate, evol, cvol, 't0')

        oe.dT.t0_1 = bf.compute_days(self.indata.date_t0, self.indata.date_tf)  # time interval in days from beginning of subset to T1
        oe.qhat = self.indata.q  # set theoretical rate of eruptions (m3/day)

        # save inside class (might modify and delete later)
        self.oe = oe

        return self.oe

    # TODO DELETE ---------------------------------------------------------------
    def print_real_dataset(self):
        """Print info about the period of real data to be used for prediction"""

        print('...')
        bf.print_period(self.date_t0, self.date_t1)
        bf.print_n_eruptions(self.n)
        bf.print_vol_stats(self.evol_mean, self.evol_std, self.evol_sum)
        bf.print_cvol(self.cvol_t0, self.cvol_t1)
        bf.print_time(self.dT_mean, self.dT_std, self.time_total)
        bf.print_rate(self.qhat)

        #         bf.print_mark()
        #
    # TODO END DELETE ------------------------------------------------------------

    def run_methods(self):

        # print real if available
        self.oe.print_instance(0)

        # deterministic method (2)
        self.deterministic_method()
        self.oe.print_instance(2)

        # stochastic method (3)
        self.stochastic_method()
        self.oe.print_instance(3)

        return self.oe


    # ------------------------------------------------------------
    # METHOD 2: DETERMINISTIC
    def deterministic_method(self):
        """Set the theoretical rate of eruptions (m3/day) for deterministic method"""

        # get parameters for deterministic method
        cvolT1, q, dT = self.oe.get_parameters(method=2)

        if q is None:
            print("No theoretical rate of eruptions (q) set. Please use set_qperiod() to set it.")
            return

        # cumulative volume at T2
        cvolT2 = bf.state_equation(cvolT1, q, dT)

        # save
        self.oe.save_result(cvolT2, dT, method=2)

    # METHOD 3: STOCHASTIC FORECAST
    def stochastic_method(self):
        """Set the theoretical rate of eruptions (m3/day) for stochastic method"""

        # get parameters for stochastic method
        cvolT1, qhat, N = self.oe.get_parameters(method=3)

        # set up simulation parameters - time interval
        dTdata = self.indata.intervals  # use dT from historical data
        if len(dTdata) == 0:
            dTdata = np.random.randint(2, 1200, 50)  # if no data, use random dT

        dTsim = np.random.choice(dTdata, N, replace=True)

        # compute CVOL(T2) = CVOL(T1) + Qhat * dTsim (for each dT)
        CVsim = [bf.state_equation(cvolT1, qhat, dT) for dT in dTsim]

        # save simulation results
        self.oe.save_result(CVsim, dTsim, method=3)



