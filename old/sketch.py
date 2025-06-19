# print period and number of eruptions
    print('==================================================')
    print(f"Cvol list: {cvol}")
    print(f"Edates list: {edates}")

    # plot eruption volumes
    dt_days = mp.plot_data(edates, evol)
    q_est, timeline = mp.plot_cvol(edates, evol, cvol)

    # print period and number of eruptions
    print(f"Intervals between eruptions (days): {dt_days}")
    print(f"Timeline of eruptions: {timeline}")
    print('==================================================')

    # ------------------------------------------------------------
    # ESTIMATION: NON-PARAMETRIC UNCERTAINTY PROPAGATION
    #
    # ------------------------------------------------------------
    t1 = timeline[-1]
    cv1 = cvol[-1]
    qhat = q_est
    # time intervals (sampling) - 10000 seems to be enough
    N = 10000
    dT_sim = np.random.choice(dt_days, N, replace=True)
    # compute cv(t2) = cv(t1) + qhat * dT_sim for each dT
    cv2_sim = cv1 + qhat * dT_sim
    # compute mean and std of cv2_sim
    mean_cv2 = np.mean(cv2_sim)
    std_cv2 = np.std(cv2_sim)
    lower, upper = np.percentile(cv2_sim, [2.5, 97.5])
    lower_, upper_ = stats.t.interval(0.95, N-1, loc=mean_cv2, scale=std_cv2/np.sqrt(N))

    evol_mean = bf.compute_delta_vol(cv1, mean_cv2)
    evol_lower = bf.compute_delta_vol(cv1, lower)
    evol_upper = bf.compute_delta_vol(cv1, upper)

    print(f"---- Mean cumulative volume at T2: {mean_cv2:.0f} m3 (evol(T2) = {evol_mean:.0f} m3)")
    print(f"95% CI: [{lower:.0f}, {upper:.0f}] m3 (or {evol_lower:.0f}, {evol_upper:.0f} m3)")

    # find most likely value
    # Make bins
    bins = np.arange(cv2_sim.min(), cv2_sim.max() + 2)
    # Compute histogram
    h, _ = np.histogram(cv2_sim, bins)
    # Find most frequent value
    mode = bins[h.argmax()]
    evol_mode = bf.compute_delta_vol(cv1, mode)
    print(f"Mode cumulative volume at T2: {mode:.0f} m3 (evol(T2) = {evol_mode:.0f} m3)")

    # find most likely value for T2
    print(f"Mean time interval (simulated) : {np.mean(dT_sim):.0f} days after T1")

    plt.hist(cv2_sim, bins=50, color='cornflowerblue', edgecolor='black', alpha=0.75)
    plt.axvline(mean_cv2, color='k', linestyle='--', label='Mean prediction')
    plt.axvline(lower, color='r', linestyle=':', label='95% CI')
    plt.axvline(upper, color='r', linestyle=':')
    plt.title('Non-parametric Prediction of Volume at T2')
    plt.xlabel('Cumulative Volume at T2')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # TIME
    # compute mean and std of cv2_sim
    mean_t2 = np.mean(dT_sim)
    std_T = np.std(dT_sim)
    lowerT, upperT = np.percentile(dT_sim, [2.5, 97.5])
    print(f"Mean T2: {mean_t2:.0f} days after T1")
    print(f"95% CI: [{lowerT:.0f}, {upperT:.0f}] days after T1")

    plt.hist(dT_sim, bins=50, color='cornflowerblue', edgecolor='black', alpha=0.75)
    plt.axvline(mean_t2, color='k', linestyle='--', label='Mean prediction')
    plt.axvline(lowerT, color='r', linestyle=':', label='95% CI')
    plt.axvline(upperT, color='r', linestyle=':')
    plt.title('Non-parametric Prediction of Time Interval')
    plt.xlabel('$\Delta$T = T$_2$ - T$_1$')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_rate_stats(self):
    time_period_days = (self.list_date[-1] - self.list_date[0]).days
    time_period_years = self.list_date[-1].year - self.list_date[0].year  # approximate conversion to years
    # sanity check
    print(f"Period analyzed: {self.list_date[0]} - {self.list_date[-1]}")
    print(f"Time period: {time_period_days} days ({time_period_years} years)")
    print(f"Number of eruptions: {self.n}")
    print("...Time between between eruptions (dt in days):")
    print(f"From all period: tf - t0 / n \n> dt_all = {time_period_days / self.n:.2f} days")
    print(f"Mean adjusted with each new eruption\n> dt_iter = {self.mean_dt:.2f} +- {self.std_dt:.2f} days")

    print(f"...Rate of eruptions (Q in km3/year):")
    if self.Qlong is not None:
        print(f"Long-term: Q_long = {self.Qlong:.4f} km3/year")
    else:
        print("Long-term rate of eruptions not set.")

    print(f"From all period: [Vcum(tf) - Vcum(t0)] / dT \n>> Q_all = {self.list_Q[-1]:.4f} km3/year")
    print(f"Mean of Q adjusted with each new eruption \n>> Q_iter = {self.mean_Q:.5f} +- {self.std_Q:.5f} km3/year")


# from collect data
    # some other computation (might delete later)
    self.list_dt = bf.compute_intervals(self.list_date)
    self.timeline = bf.compute_timeline(self.list_dt, 0)  # start from 1

    if self.printing:
        # all dataframe
        print('Volcano df ', self.df_volcano)
        print('Cum Vol ', self.list_cumvol)
        # dates
        print('dt in days: ', self.list_dt)


    def plot_volume(self):
        """Plot the cumulative volume of the volcano and date the data was collected"""

        title, xlabel, ylabel, mylegend = self.get_plot_title()
        fig, ax = plt.subplots()


        # convert date to string for plotting
        xvalues = self.timeline
        yvalues = self.list_cumvol

        # PLOT
        # cumulative volume
        ax.plot(xvalues, yvalues, linestyle='dashed', color='b', linewidth=0.5, marker='.', markersize=10, mfc='red', label=mylegend[0])
        # erupted volume
        yvalues2 = self.list_eruptvol
        ax.plot(xvalues, yvalues2 , linestyle='dashed', color='g', linewidth=0.5, marker='.', markersize=10,
                mfc='red', label=mylegend[1])

        # set limits for x and y axes
        th = 1e8
        ax.set(ylim=(min(yvalues) - th, max(yvalues) + th))
        # title and labels
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        # grid
        ax.grid()
        ax.set_xticks(xvalues[::1])
        ax.set_xticklabels(xvalues[::1], rotation=45, ha='right', fontsize=8)

        # show
        plt.show()
        # save the figure
        save_path = os.path.join(self.current_dir, self.plot_dir)
        fig.savefig(save_path + "/piton-cumvol.png")


   def piton_periods(self):
        """ Useless, todo DELETE    """
        start_p1 = pd.Timestamp('1936-08-01')
        end_p1 = pd.Timestamp('1998-03-11')
        start_p2 = pd.Timestamp('1999-07-19')

        r1, rend = 0, 0

        if self.list_date[-1] < start_p2:
            # if the last date is before the start of period 2
            self.period = 1
            r1, rend = 1, 74
        else:
            if self.list_date[0] >= start_p2:
                # if the first date is after the start of period 2
                self.period = 2
                r1, rend = 74, 120
            else:
                # if the data is from both periods
                self.period = 0
                r1, rend = 1, 120

        self.r1, self.rend = r1, rend
        return r1, rend

# -=====================================================

    # METHOD 3: STOCHASTIC
    def stochastic_method(self):

        return


    def set_qtheory(self, qtheory: float):
        """Set the theoretical rate of eruptions (m3/day)"""
        self.qperiod = qtheory

    def run_prediction_methods2(self):

        # stochastic forecast
        self.one_step_ahead()
        # choose estimate
        self.choose_estimate()
        self.compute_stats()
        # error
        self.forecast_error()

        # deterministic forecast
        self.deterministic()

        # PRINT
        bf.print_submark()
        # print real
        bf.print_one_eruption(self.next_id, self.real_evol_t2, self.real_cvol_t2,
                              self.real_date_t2, self.real_dT)

        # print results
        bf.print_deterministic(self.evolT2_det, self.cvolT2_det)

        bf.print_deterministic_error(self.error_evolT2_det, self.error_cvolT2_det,
                                     self.error_evolT2_det_per, self.error_cvolT2_det_per)


        # print results
        bf.print_prediction(self.evolT2_hat, self.cvolT2_hat, self.dT_hat,
                            (self.cvolT2_lower, self.cvolT2_upper))


        # print error
        bf.print_prediction_error(self.error_evolT2, self.error_cvolT2,
                                  self.error_evol_per, self.error_cvol_per, self.error_dT2)

        bf.print_mark()



    # ESTIMATION: NON-PARAMETRIC UNCERTAINTY PROPAGATION
    # (stochastic forecast)
    def one_step_ahead(self):
        """Predict NEXT eruption
        What I need for estimation:
        qhat, DT_list (days), T1 (timeline, day)
        and cvol(t1) = vcol[-1]"""

        # current cumulative volume at T1
        CV1 = [self.cvol_t1] * self.N
        dTdata = self.dT_days
        N = self.N
        qhat = self.qhat

        # ------------------------------------ SIMULATION
        # sampling time intervals
        # TODO change this for normal skewed?
        dTsim = np.random.choice(dTdata, N, replace=True)

        # compute CVOL(T2) = CVOL(T1) + Qhat * dTsim (for each dT)
        CV2 = CV1 + qhat * dTsim
        # ------------------------------------

        # save simulated data
        self.sim_cvolT2 = CV2
        self.sim_dT = dTsim

    def compute_stats(self):
        # STATS
        # compute mean and std of cv2_sim
        self.cvolT2_mean = np.mean(self.sim_cvolT2)
        self.cvolT2_std = np.std(self.sim_cvolT2)
        self.cvolT2_lower, self.cvolT2_upper = np.percentile(self.sim_cvolT2, [2.5, 97.5])

        # compute mean and std of dTsim
        self.dTsim_mean = np.mean(self.sim_dT)
        self.dTsim_std = np.std(self.sim_dT)
        self.dTsim_lower, self.dTsim_upper = np.percentile(self.sim_dT, [2.5, 97.5])

    def choose_estimate(self, best='median'):
        """Choose estimate, for now use mean or median of dTsim and cvolT2"""

        if best == 'mean':
            # choose mean
            CV2hat = np.mean(self.sim_cvolT2)
            dThat = np.mean(self.sim_dT)
        elif best == 'median':
            # choose median
            CV2hat = np.median(self.sim_cvolT2)
            dThat = np.median(self.sim_dT)
        else:
            # if no best is specified, use mean (for now)
            CV2hat = np.mean(self.sim_cvolT2)
            dThat = np.mean(self.sim_dT)

        # save estimates
        self.cvolT2_hat = CV2hat
        self.dT_hat = dThat  # next eruption time interval
        # next eruption volume
        self.evolT2_hat = self.cvolT2_hat - self.cvol_t1

    def forecast_error(self):
        """Compute forecast error between predicted and real data"""

        if self.real_dT is None or self.real_evol_t2 is None or self.real_cvol_t2 is None:
            print("Real next eruption data not set. Please use add_real_next() to set it.")
            return

        # compute ERROR
        self.error_dT2, _ = bf.compute_error(self.dT_hat, self.real_dT)
        self.error_evolT2, self.error_evol_per = bf.compute_error(self.evolT2_hat, self.real_evol_t2)
        self.error_cvolT2, self.error_cvol_per = bf.compute_error(self.cvolT2_hat, self.real_cvol_t2)



    # TODO compare error between methods
    # TO BE CALLED IN PLOTS: call this class IN PLOTS to plot stuff


bf.print_mark()
print(f"Error EVOL(t2) %: \nMEAN: {np.mean(error_evol):.1f} | MAX {max(error_evol):.1f}| MIN {min(error_evol):.1f} ")
j = start_after_eruption
for er in error_evol:
    j += 1
    print(f"({j}) {er:.2f} %", end=" | ")