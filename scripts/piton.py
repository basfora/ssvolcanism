"""Place-holder for data specific scripts and plot generation"""

from classes.collectdata import VolcanoData as vd
from classes.computethings import ComputeThings as ct
from classes.basicfun import basicfun as bf
from classes.myplots import MyPlots as mp

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# compute stats for Piton de la Fournaise volcano, period 1 (1936-1998)

if __name__ == '__main__':

    name_file = 'PitondelaFournaise_data'

    # import data from Excel file >> Piton de la Fournaise
    piton_data = vd(name=name_file, printing=False)
    # get data from the file
    # period 1: 1 to 74 | period 2: 74 to 120
    # let's try to estimate eruptions 69 - 72, r1 = 1, rend = 69
    r1, rend = 1, 6
    piton_data.get_data(r1, rend)

    # whole period data
    idxf = piton_data.n - 1
    edates, evol, cvol = piton_data.output_rel_data(0, piton_data.n)

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
    # what I need for estimation: qhat, DT_list (days), T1 (timeline, day) and cvol(t1) = vcol[-1]
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

    evol_mean = bf.compute_evol(cv1, mean_cv2)
    evol_lower = bf.compute_evol(cv1, lower)
    evol_upper = bf.compute_evol(cv1, upper)

    print(f"---- Mean cumulative volume at T2: {mean_cv2:.0f} m3 (evol(T2) = {evol_mean:.0f} m3)")
    print(f"95% CI: [{lower:.0f}, {upper:.0f}] m3 (or {evol_lower:.0f}, {evol_upper:.0f} m3)")

    # find most likely value
    # Make bins
    bins = np.arange(cv2_sim.min(), cv2_sim.max() + 2)
    # Compute histogram
    h, _ = np.histogram(cv2_sim, bins)
    # Find most frequent value
    mode = bins[h.argmax()]
    evol_mode = bf.compute_evol(cv1, mode)
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

















