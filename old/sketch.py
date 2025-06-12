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
    if self.Q_long is not None:
        print(f"Long-term: Q_long = {self.Q_long:.4f} km3/year")
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
