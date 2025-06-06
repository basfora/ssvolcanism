"""Data used for analysis and prediction
One instance for each prediction pt"""


class MyPrediction:

    def __init__(self, edates: list, evol: list, cvol: list, idx=None):

        # for storage and information purposes
        # inputs (real data collection)
        self.in_edates = edates
        self.in_evol = evol
        self.in_cvol = cvol
        self.n = 0

        # actual data for prediction
        self.real_next_date = None
        self.real_next_evol = None
        self.real_next_cvol = None

        # period being used for prediction
        self.date_t0 = None
        self.date_tf = None

        # to be used in analysis
        # basic stats to use in prediction
        self.mean_evol = 0.0

        # cumulative volume
        self.cvol_t0 = 0.0
        self.cvol_tf = 0.0
        self.cvol_next = 0.0


        # save real data (if idx is none, save all)

        # get and print period info

        # compute intervals

        # prediction (stochastic forecast)

        # compare with real

        # output results and error

        # TO BE CALLED IN PLOTS: use this class to plot stuff






