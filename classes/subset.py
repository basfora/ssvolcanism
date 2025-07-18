"""Historical Data from a volcano"""


import datetime

from classes.basicfun import Basicfun as bf
import numpy as np

# todo: I need to add a class for period info (Qline)
# todo need to have a class for a datasubset to interface with collect data AND prediction without having to repeat
class MySubset:
    """Info about period limits"""

    def __init__(self, edates: list, evol, cvol):

        # idenfification purposes
        self.volcano = None

        # is it a period of interest?
        self.period = False  # True if period is of interest, False if just some subset

        # period characteristics IN THE FILE: periods, init/final eruption ID, init/final dates, Q

        # TODO FOLLOW MY PREVIOUS CONVENTION!! (prediction / eruptions naming convention)
        # first and last eruptions
        self.first_eid = None
        self.last_eid = None

        # start and end dates
        self.start_edate = None
        self.end_edate = None

        self.date0 = None
        self.datef = None

        # rate for that period
        self.q = None  # m3/day
        self.q_yr = None  # km3/yr

        # data from the file (add second)
        self.edates = edates
        self.evol = evol
        self.cvol = cvol

        # computed or extracted from data
        # start and end cumulative volume
        self.cvol_t0 = None
        self.cvol_tf = None

        # todo put other stuff here? intervals, timeline, etc.

    # TODO
    # info clearly labeled on the data file: PERIODS, INIT/FINAL ERUPTION ID, INIT/FINAL DATES, Q
    def set_dates(self, date0, datef):
        """Set the start and end dates of the period"""
        if isinstance(date0, datetime.date) and isinstance(datef, datetime.date):
            self.date0 = date0
            self.datef = datef
        else:
            exit("Invalid date format. Use datetime.date objects.")

    def set_name(self, name):

        """Set the name of the period"""
        if 'piton' in name.lower():
            self.name = 'Piton de la Fournaise'
        elif 'hawaii' in name.lower():
            self.name = 'Hawaii'
        elif 'iceland' in name.lower():
            self.name = 'Iceland'
        elif 'galapagos' in name.lower():
            self.name = 'Western Galapagos'
        else:
            exit(f"Unknown volcano name: {name}. Please check the name and try again.")

