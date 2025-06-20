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
        bf.print_estimate(self.evolT2_det, self.cvolT2_det)

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


    # OBSOLETE PLOTS
    def plot_set02(self, eruptions: list, savename=None):
        """Plot Cumulative Volume (CVOL) real and deterministic prediction"""

        self.period = bf.format_period(eruptions[0].date.real, eruptions[-1].date.real)

        suptitle = f"{self.volcano_name} - {self.period}"
        fig, ax = plt.subplots(1, 1, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # separate eruptions (class)
        ep1 = eruptions[:73]  # period I
        ep2 = eruptions[73:]  # period II

        # sanity check (date and q used)
        assert len(ep1) == 73, "Period I should have 73 eruptions"
        assert len(ep2) == 46, "Period II should have 46 eruptions"
        assert ep1[0].date.real == datetime.date(1936, 8, 1), "First date of period I should be 1936-1-8"
        assert ep1[-1].date.real == datetime.date(1998, 3, 11), "Last date of period I should be 1998-03-11"
        assert round(ep1[1].q_period, 4) == round(ep1[-1].q_period, 4) == round(bf.Qy_to_Qday(0.0107),
                                                                                4), "Q for period I should be 0.0107 km3/yr"

        assert ep2[0].date.real == datetime.date(1999, 7, 19), "First date of period II should be 1999-07-19"
        assert ep2[-1].date.real == datetime.date(2018, 7, 13), "First date of period II should be 2018-07-13"
        assert round(ep2[0].q_period, 4) == round(ep2[-1].q_period, 4) == round(bf.Qy_to_Qday(0.0228),
                                                                                4), "Q for period II should be 0.0228 km3/yr"

        # -------------------- DATA PREPARATION
        # xvalues: dates of eruptions
        xvalues = [e.date.real for e in eruptions]  # n
        # yvalues: real cumulative volumes (CVOL)
        yvalues_real = [e.cvol.real for e in eruptions]  # n

        # expected values (deterministic prediction)
        xvalues1 = [e.date.real for e in ep1[1:]]  # skip first eruption, n-1
        yvalues1 = [e.cvol.det.value for e in ep1 if e.cvol.det.value is not None]  # n-1
        xvalues2 = [e.date.real for e in ep2]  # skip first eruption, n-1
        yvalues2 = [e.cvol.det.value for e in ep2 if e.cvol.det.value is not None]  # n-1

        # ---------------- PLOT 1 (MAIN)
        # Plot real values (n)
        ax.scatter(xvalues, yvalues_real, marker=self.marker_real, color=self.color_real, linewidth=3,
                   label=self.leg_real)
        # Plot predicted values - period I (eruptions # 2 - 73)
        ax.scatter(xvalues1, yvalues1, marker=self.marker_det, color=self.color_pred1, linewidth=1, label=self.leg_Q1)
        # Plot predicted values - period II (eruptions # 74 - end)
        ax.scatter(xvalues2, yvalues2, marker=self.marker_det, color=self.color_pred2, linewidth=1, label=self.leg_Q2)
        # plot title, labels and legend
        mytitle = f"{self.title_cvol}: {self.title_exp}"
        ax.set(title=mytitle, xlabel=self.label_date, ylabel=self.label_vol)
        ax.legend(frameon=False, loc='upper left')
        ax.grid(True)

        # set limits for x and y axes
        ylim = max(yvalues_real) * self.yth  # threshold for y-axis
        ax.set(ylim=(min(yvalues_real) - 2 * ylim, max(yvalues_real) + ylim))

        # show, save and close
        plt.show()
        if savename is None:
            savename = 'cvol'
        self.save_fig(fig, savename)

    def plot_set03(self, eruptions: list, savename=None):
        """Error between real and predicted CVOL, and error histogram"""

        self.period = bf.format_period(eruptions[0].date.real, eruptions[-1].date.real)

        suptitle = f"{self.volcano_name} {self.title_cvol} \n{self.period}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # ------------------------------------- PLOT 2: ERROR
        xvalues = [e.date.real for e in eruptions if e.id > 1]  # n-1
        yvalues = [e.cvol.det.error for e in eruptions if e.cvol.det.value is not None]  # n-1

        # add stats: mean and std
        error_mean, error_std = bf.compute_mean_std(yvalues)
        error_median = bf.m3_to_km3(np.median(yvalues))
        e_mean_km3 = bf.m3_to_km3(error_mean)
        e_std_km3 = bf.m3_to_km3(error_std)

        # PLOT 1 (LEFT) - error pts and mean/std lines

        # Plot error between real and predicted CVOL
        ax[0].scatter(xvalues, yvalues, marker=self.marker_error, color=self.color_error, linewidth=2,
                      label=self.leg_error)
        # extra stats
        ax[0].axhline(error_mean, color=self.color_mean, linestyle=self.line_mean, linewidth=2,
                      label=f"{self.label_mean_symbol} {e_mean_km3:.4f} km$^3$")
        ax[0].axhline(error_mean + error_std, color=self.color_std, linestyle=self.line_std,
                      label=f"$\pm \sigma$ = {e_std_km3:.4f} km$^3$")
        ax[0].axhline(error_mean - error_std, color=self.color_std, linestyle=self.line_std)
        # median
        # ax[0].axhline(error_median, color=self.color_median, linestyle=self.line_mean,
        #              label=f"{self.label_median} km$^3$")

        # plot title, labels and legend
        ax[0].set(title=self.title_error, xlabel=self.label_date, ylabel=self.label_vol)

        ax[0].legend(frameon=False, loc='upper left')
        ax[0].grid(True)

        # set limits for x and y axes
        ylim = max(yvalues) * self.yth
        ax[0].set(ylim=(min(yvalues) - 2 * ylim, max(yvalues) + ylim))

        # ------------------------------------- PLOT 2 (RIGHT) - ERROR HISTOGRAM
        # Plot histogram with KDE

        sns.histplot(yvalues, kde=True, ax=ax[1], color=self.color_hist, bins=20)

        # plot identifiers
        plt.title(self.title_error_hist)
        # plot stats
        plt.axvline(error_mean, color=self.color_mean, linestyle=self.line_mean,
                    label=f"{self.label_mean_symbol} {e_mean_km3:.4f} km$^3$")
        plt.axvline(error_mean + error_std, color=self.color_std, linestyle=self.line_std,
                    label=f"$\pm \sigma$ = {e_std_km3:.4f} km$^3$")
        plt.axvline(error_mean - error_std, color=self.color_std, linestyle=self.line_std)

        plt.xlabel(self.label_vol)
        plt.ylabel(self.label_freq)
        plt.legend(frameon=False)

        # printout error stats
        print('CVOL Error Statistics:')
        print(f"Mean: {e_mean_km3:.4f} +- {e_std_km3:.4f} km3")
        print(f"Median: {error_median:.4f} km3")
        print(f"Min: {bf.m3_to_km3(min(yvalues)):.4f} | Max: {bf.m3_to_km3(max(yvalues)):.4f} km3")
        # ----------------

        # show, save and close
        plt.show()
        if savename is None:
            savename = 'cvol'
        self.save_fig(fig, savename)

        return

    def plot_set04(self, eruptions: list, savename=None):
        """Plot Eruption Volume (EVOL) real and deterministic prediction"""

        self.period = bf.format_period(eruptions[0].date.real, eruptions[-1].date.real)

        suptitle = f"{self.volcano_name} - {self.period}"
        fig, ax = plt.subplots(1, 1, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # separate eruptions (class)
        ep1 = eruptions[:73]  # period I
        ep2 = eruptions[73:]  # period II

        # sanity check (date and q used)
        assert len(ep1) == 73, "Period I should have 73 eruptions"
        assert len(ep2) == 46, "Period II should have 46 eruptions"
        assert ep1[0].date.real == datetime.date(1936, 8, 1), "First date of period I should be 1936-1-8"
        assert ep1[-1].date.real == datetime.date(1998, 3, 11), "Last date of period I should be 1998-03-11"
        assert round(ep1[1].q_period, 4) == round(ep1[-1].q_period, 4) == round(bf.Qy_to_Qday(0.0107),
                                                                                4), "Q for period I should be 0.0107 km3/yr"

        assert ep2[0].date.real == datetime.date(1999, 7, 19), "First date of period II should be 1999-07-19"
        assert ep2[-1].date.real == datetime.date(2018, 7, 13), "First date of period II should be 2018-07-13"
        assert round(ep2[0].q_period, 4) == round(ep2[-1].q_period, 4) == round(bf.Qy_to_Qday(0.0228),
                                                                                4), "Q for period II should be 0.0228 km3/yr"

        # -------------------- DATA PREPARATION
        # xvalues: dates of eruptions
        xvalues = [e.date.real for e in eruptions]  # n
        # yvalues: real cumulative volumes (CVOL)
        yvalues_real = [e.evol.real for e in eruptions]  # n

        # expected values (deterministic prediction)
        xvalues1 = [e.date.real for e in ep1[1:]]  # skip first eruption, n-1
        yvalues1 = [e.evol.det.value for e in ep1 if e.evol.det.value is not None]  # n-1
        xvalues2 = [e.date.real for e in ep2]  # skip first eruption, n-1
        yvalues2 = [e.evol.det.value for e in ep2 if e.evol.det.value is not None]  # n-1

        # ---------------- PLOT 1 (MAIN)
        # Plot real values (n)
        ax.scatter(xvalues, yvalues_real, marker=self.marker_real, color=self.color_real, linewidth=3,
                   label=self.leg_real)
        # Plot predicted values - period I (eruptions # 2 - 73)
        ax.scatter(xvalues1, yvalues1, marker=self.marker_det, color=self.color_pred1, linewidth=1, label=self.leg_Q1)
        # Plot predicted values - period II (eruptions # 74 - end)
        ax.scatter(xvalues2, yvalues2, marker=self.marker_det, color=self.color_pred2, linewidth=1, label=self.leg_Q2)
        # plot title, labels and legend
        mytitle = f"{self.title_evol}: {self.title_exp}"
        ax.set(title=mytitle, xlabel=self.label_date, ylabel=self.label_vol)
        ax.legend(frameon=False, loc='upper left')
        ax.grid(True)

        # set limits for x and y axes
        ylim = max(yvalues_real) * self.yth  # threshold for y-axis
        ax.set(ylim=(min(yvalues_real) - 2 * ylim, max(yvalues_real) + ylim))

        # show, save and close
        plt.show()
        if savename is None:
            savename = 'cvol'
        self.save_fig(fig, savename)

    def plot_set05(self, eruptions: list, savename=None):
        """Plot cumulative volume (CVOL) real and stochastic prediction, and error histogram"""

        self.period = bf.format_period(eruptions[0].date.real, eruptions[-1].date.real)

        suptitle = f"{self.volcano_name} {self.title_cvol} \n{self.period}"
        fig, ax = plt.subplots(1, 2, figsize=(self.width, self.height))
        fig.suptitle(suptitle)

        # ------------------------------------- PLOT 2: ERROR
        xvalues = [e.date.real for e in eruptions if e.id > 1]  # n-1
        yvalues = [e.evol.det.error for e in eruptions if e.evol.det.value is not None]  # n-1

        # add stats: mean and std
        error_mean, error_std = bf.compute_mean_std(yvalues)
        error_median = bf.m3_to_km3(np.median(yvalues))
        e_mean_km3 = bf.m3_to_km3(error_mean)
        e_std_km3 = bf.m3_to_km3(error_std)

        # PLOT 1 (LEFT) - error pts and mean/std lines

        # Plot error between real and predicted CVOL
        ax[0].scatter(xvalues, yvalues, marker=self.marker_error, color=self.color_error, linewidth=2,
                      label=self.leg_error)
        # extra stats
        ax[0].axhline(error_mean, color=self.color_mean, linestyle=self.line_mean, linewidth=2,
                      label=f"{self.label_mean_symbol} {e_mean_km3:.4f} km$^3$")
        ax[0].axhline(error_mean + error_std, color=self.color_std, linestyle=self.line_std,
                      label=f"$\pm \sigma$ = {e_std_km3:.4f} km$^3$")
        ax[0].axhline(error_mean - error_std, color=self.color_std, linestyle=self.line_std)
        # median
        # ax[0].axhline(error_median, color=self.color_median, linestyle=self.line_mean,
        #              label=f"{self.label_median} km$^3$")

        # plot title, labels and legend
        ax[0].set(title=self.title_error, xlabel=self.label_date, ylabel=self.label_vol)

        ax[0].legend(frameon=False, loc='upper left')
        ax[0].grid(True)

        # set limits for x and y axes
        ylim = max(yvalues) * self.yth
        ax[0].set(ylim=(min(yvalues) - 2 * ylim, max(yvalues) + ylim))

        # ------------------------------------- PLOT 2 (RIGHT) - ERROR HISTOGRAM
        # Plot histogram with KDE

        sns.histplot(yvalues, kde=True, ax=ax[1], color=self.color_hist, bins=20)

        # plot identifiers
        plt.title(self.title_error_hist)
        # plot stats
        plt.axvline(error_mean, color=self.color_mean, linestyle=self.line_mean,
                    label=f"{self.label_mean_symbol} {e_mean_km3:.4f} km$^3$")
        plt.axvline(error_mean + error_std, color=self.color_std, linestyle=self.line_std,
                    label=f"$\pm \sigma$ = {e_std_km3:.4f} km$^3$")
        plt.axvline(error_mean - error_std, color=self.color_std, linestyle=self.line_std)

        plt.xlabel(self.label_vol)
        plt.ylabel(self.label_freq)
        plt.legend(frameon=False)

        # printout error stats
        print('EVOL Error Statistics:')
        print(f"Mean: {e_mean_km3:.4f} +- {e_std_km3:.4f} km3")
        print(f"Median: {error_median:.4f} km3")
        print(f"Min: {bf.m3_to_km3(min(yvalues)):.4f} | Max: {bf.m3_to_km3(max(yvalues)):.4f} km3")
        # ----------------

        # show, save and close
        plt.show()
        if savename is None:
            savename = 'cvol'
        self.save_fig(fig, savename)

        return

    @staticmethod
    def plot_eruptions(dates, volumes, cumulative_volumes):
        """Plot eruption volumes over time."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=dates, y=volumes, label='Eruption Volume (m3)', marker='o')
        sns.lineplot(x=dates, y=cumulative_volumes, label='Cumulative Volume (m3)', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Volume (m3)')
        plt.title('Eruption Volumes Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_data(dates: list, values: list, cvalues=None):

        """Plot data with histograms and time series."""

        plt.figure(figsize=(12, 12))
        nplots = 2

        # VOLUME ERUPTIONS (EVOL)
        n = len(dates)
        # mean and std
        mean_v, std_v = bf.compute_mean_std(values)
        mean_values = [mean_v] * n

        print(f"Number of eruptions: {n}")
        t0, tf = bf.print_period(dates[0], dates[-1])
        print(f"Volume per eruption (m3):\n Mean: {mean_v:.0f} \n Std Dev: {std_v:.0f}")

        if cvalues is not None:
            q = bf.compute_q(cvalues[0], cvalues[-1], (dates[-1] - dates[0]).days)
            e_line = [q * (dates[i] - dates[i-1]).days for i in range(1, n)]

        # eruptions and volumes
        title1 = 'Eruptions\n' + t0 + ' to ' + tf
        label1 = 'Volume ($m^3$)'
        title2 = 'Volume '

        plt.subplot(nplots, nplots, 1)
        plt.scatter(dates, values, marker='x', color='b', label='real data')
        plt.plot(dates, mean_values, color='k', linestyle='--')
        if cvalues is not None:
            #plt.plot(dates[1:], e_line, color='g', linestyle='--', label='Q line')
            plt.scatter(dates[1:], e_line, marker='x', color='r', label='Q * $\Delta T_{k}$')
            plt.legend()
        plt.title(title1)
        plt.xlabel("Date of Eruption")
        plt.ylabel(label1)

        plt.subplot(nplots, nplots, 2)
        sns.histplot(values, kde=True)
        plt.title("Distribution " + title2)
        plt.xlabel(label1)
        plt.ylabel("Frequency")

        # TIME INTERVALS (DT)
        values_dt_days = bf.compute_intervals(dates)
        aux_x = [i for i in range(n - 1)]
        title3 = 'Time Intervals '
        xlabel3 = 'Eruption Number'
        label3 = '$\Delta~T$ (days)'

        # mean and std
        mean_dt, std_dt = bf.compute_mean_std(values_dt_days)
        mean_dt_values = [mean_dt] * (n-1)

        print(f"Time interval between eruptions (days): \n Mean: {mean_dt:.0f} \n Std Dev: {std_dt:.0f}")

        plt.subplot(nplots, nplots, 3)
        plt.scatter(aux_x, values_dt_days, marker='x', color='b')
        plt.plot(aux_x, mean_dt_values, color='k', linestyle='--', label='Mean Interval')
        plt.title(title3)
        plt.xlabel(xlabel3)
        plt.ylabel(label3)

        plt.subplot(nplots, nplots, 4)
        sns.histplot(values_dt_days, kde=True)
        plt.title("Distribution " + title3)
        plt.xlabel(label3)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

        return values_dt_days


    @staticmethod
    def plot_cvol(dates: list, values: list, cvalues: list):
        """Plot cumulative volume with a rate line."""

        plt.figure(figsize=(12, 12))

        q = bf.compute_q(cvalues[0], cvalues[-1], (dates[-1] - dates[0]).days)
        q_years = bf.Qday_to_Qy(q)
        print(f"Computed rate Q = {q:.4f} m3/day ({q_years:.5f} km3/year)")
        cvol_theory = bf.get_q_line(q, dates, cvalues)

        values_dt_days = bf.compute_intervals(dates)

        # CUMULATIVE VOLUME ERUPTIONS (CVOL)
        n = len(dates)

        nrow = 2
        ncol = 2

        t0, tf = bf.print_period(dates[0], dates[-1])
        title1 = 'Cumulative Volume'
        title2 = '\n' + t0 + ' to ' + tf
        label1 = "Volume ($m^3$)"
        label3 = "Error ($m^3$)"

        # FIT TO LINE
        timeline = bf.compute_timeline(values_dt_days)

        # linear squares fit
        a, b = np.polyfit(timeline, cvalues, 1)
        print(f"Linear fit: y = {a:.4f}x + {b:.4f}")
        fit_values = [a * t + b for t in timeline]

        # using computed q
        q_pred_values = [q * t + cvalues[0] for t in timeline]

        # CVOL PLOT
        plt.subplot(nrow, ncol, 1)
        myleg = 'Q = ' + str(round(q_years, 4)) +' m3/day'
        plt.scatter(dates, cvol_theory, marker='x', color='r', label=myleg)
        plt.scatter(dates, cvalues, marker='x', color='b')
        plt.title(title1 + title2)
        plt.xlabel("Date of Eruption")
        plt.ylabel(label1)
        plt.legend()

        # dt vs evol
        # plt.scatter(values_dt_days, values[1:], marker='x', color='r')
        # plt.title("Eruption Volume vs Time Interval ")
        # plt.xlabel('$\Delta T$ (days)')
        # plt.ylabel(label1)

        plt.subplot(nrow, ncol, 2)
        plt.scatter(timeline, cvol_theory, marker='x', color='r', label='$Q * \Delta T_{k} + Cvol_{k-1}$')
        plt.scatter(timeline, cvalues, marker='x', color='b', label='real data')
        plt.plot(timeline, fit_values, color='g', linestyle='--',
                 label='polyfit \n(a,b) = ({:.0f}, {:.0f})'.format(a, b))
        # plt.scatter(timeline, q_pred_values, marker='.', color='k', label='q_predicted')
        plt.title(title1)
        plt.xlabel('Timeline (days)')
        plt.ylabel(label1)
        plt.legend()

        # ERROR BETWEEN CVOL_THEORY AND CVOL_DATA
        q_error = [cvol_theory[i-1] - cvalues[i] for i in range(1, n)]
        title3 = 'Error '
        aux_x = [i for i in range(n - 1)]
        error_mean, error_std = bf.compute_mean_std(q_error)
        error_mean_values = [error_mean] * (n - 1)
        print(f"Error Cvol theory and data: {abs(error_mean):.4f} +- {error_std:.4f} m3")

        plt.subplot(nrow, ncol, 3)
        plt.scatter(dates[1:], q_error, marker='o', color='b', label='CVol Error')
        plt.plot(dates[1:], error_mean_values, color='r', linestyle='--', label='Mean Error')
        plt.title(title3 + 'Cumulative Volume: Theory vs Data')
        plt.xlabel("Date of Eruption")
        plt.ylabel(label3)
        plt.legend()

        plt.subplot(nrow, ncol, 4)
        sns.histplot(q_error, kde=True)
        plt.title("Distribution " + title3)
        plt.xlabel(label3)
        plt.ylabel("Frequency")


        plt.tight_layout()
        plt.show()

        return q, timeline


    @staticmethod
    def get_plot_title(name: str, what="evol"):
        if "Piton" in name:
            title = "Piton de la Fournaise"
            xlabel = "Collection Date (year)"
            ylabel = "Volume ($m^3$)"
            legend = ["Cumulative Volume", "Erupted Volume"]
        else:
            title = "Unknown"
            xlabel = "Collection Date (year)"
            ylabel = "Cumulative Volume ($m^3$)"
            legend = ["Cumulative Volume", "Erupted Volume"]

        return title, xlabel, ylabel, legend

