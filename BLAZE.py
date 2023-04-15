import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, AnnotationBbox


h_0 = 3
g = 9.81


@jit(nopython=True)
def T_APG(v_0, v_t):
    """
        Computes the time taken by a ballistic object to reach apogee.
        
        Parameters:
        v_0 (float): Burnout velocity (m/s).
        v_t (float): Terminal velocity (m/s).

        Returns:
        t_apg (float): Time taken to reach apogee (s).

        Assumptions:
        1. Gravity, g = 9.81 m/s^2.
        2. Burnout height, h_0 = 3 m.
        3. Burnout time, t_0 = 2 * h_0 / v_0.
    """
    g = 9.81
    t_0 = (2 * 3)/v_0
    return t_0 + (v_t / g) * np.arctan(v_0 / v_t)

@jit(nopython=True)
def H_APG(v_0, v_t):
    """
        Computes the maximum height (apogee) reached by a ballistic object.
        
        Parameters:
        v_0 (float): Burnout velocity (m/s).
        v_t (float): Terminal velocity (m/s).

        Returns:
        h_apg (float): Maximum height (apogee) reached (m).

        Assumptions:
        1. Gravity, g = 9.81 m/s^2.
        2. Burnout height, h_0 = 3 m.
    """
    h_0 = 3
    g = 9.81
    return h_0 + ((v_t**2) / (2 * g)) * np.log(1 + ((v_0 / v_t)**2))

@jit(nopython=True)
def V_END(v_0, v_t):
    """
        Computes the velocity of a ballistic object at touchdown.
        
        Parameters:
        v_0 (float): Burnout velocity (m/s).
        v_t (float): Terminal velocity (m/s).

        Returns:
        v_end (float): Velocity at touchdown (m/s).

        Assumption:
        1. Gravity, g = 9.81 m/s^2.
    """
    g = 9.81
    return -v_t * np.sqrt(1 - np.exp(-(2 * g * H_APG(v_0, v_t)) / (v_t**2)))

@jit(nopython=True)
def T_END(v_0, v_t):
    """
        Computes the total time elapsed for a ballistic object to reach touchdown.
        
        Parameters:
        v_0 (float): Burnout velocity (m/s).
        v_t (float): Terminal velocity (m/s).

        Returns:
        t_end (float): Total time elapsed to reach touchdown (s).

        Assumption:
        1. Gravity, g = 9.81 m/s^2.
    """
    g = 9.81
    return T_APG(v_0, v_t) + (v_t / g) * np.arctanh(-(V_END(v_0, v_t) / v_t))

@jit(nopython=True)
def v0_tdiff_vt(v_0, t_diff, v_t):
    """
        Used to solve for burnout velocity for a given time difference and terminal velocity.
        
        Parameters:
        v_0 (float): Burnout velocity (m/s).
        t_diff (float): Time difference (s).
        v_t (float): Terminal velocity (m/s).

        Returns:
        float: Should be zero for the correct v_0, t_diff, v_t.
    """
    return T_END(v_0, v_t) - 2 * T_APG(v_0, v_t) - t_diff

@jit(nopython=True)
def vt_hapg_v0(v_t, h_0, h_apg, v_0):
    """
        Used to solve for terminal velocity for given burnout height, apogee height, and burnout velocity.
        
        Parameters:
        v_t (float): Terminal velocity (m/s).
        h_0 (float): Burnout height (m).
        h_apg (float): Apogee height (m).
        v_0 (float): Burnout velocity (m/s).

        Returns:
        float: Should be zero for the correct v_t, h_0, h_apg, v_0.
    """
    g = 9.81
    return h_0 - h_apg + ((v_t**2) / (2 * g)) * np.log(1 + ((v_0 / v_t)**2))


@jit(nopython=True)
def vt_tapg_v0(v_t, t_apg, v_0):
    """
        Used to solve for terminal velocity for given time to apogee and burnout velocity.
        
        Parameters:
        v_t (float): Terminal velocity (m/s).
        t_apg (float): Time to apogee (s).
        v_0 (float): Burnout velocity (m/s).

        Returns:
        float: Should be zero for the correct v_t, t_apg, v_0.

        Assumption:
        Time to burnout, t_0 = 2 * h_0 / v_0, where h_0 = 3 m.
    """
    g = 9.81
    t_0 = (2 * 3)/v_0
    return t_0 - t_apg + (v_t / g) * np.arctan(v_0 / v_t)


def draw_v0_vt(v_0, v_t, axis):

    axis.plot(np.linspace(0, v_0, 100), [v_t]*100, color="springgreen", linewidth = 6.0, zorder=2)
    axis.plot([v_0]*100, np.linspace(0, v_t, 100), color="springgreen", linewidth = 6.0, zorder=2)
    axis.scatter([v_0], [v_t], s=250, c="limegreen", edgecolors="black", linewidths=2, zorder=3)

    xy1 = (v_0+8, 7.5)
    offsetbox = TextArea(f"{np.round(v_0, 2)}", textprops=dict(color="limegreen", size=18, weight='bold'))
    ab1 = AnnotationBbox(offsetbox, xy1,
                        bboxprops=dict(color='white',edgecolor='black',linewidth=2.0))
    axis.add_artist(ab1)

    xy2 = (22.5, v_t+2)
    offsetbox = TextArea(f"{np.round(v_t, 2)}", textprops=dict(color="limegreen", size=18, weight='bold'))
    ab2 = AnnotationBbox(offsetbox, xy2,
                        bboxprops=dict(color='white',edgecolor='black',linewidth=2.0))
    axis.add_artist(ab2)

    

def plot_tdiff(v_t, t_diff, axis):

    y_min = v_t
    y_max = 65.0
    y_vt = np.linspace(y_min, y_max, 100)
    x_v0 = np.array([fsolve(v0_tdiff_vt, 0.1, args=(t_diff, y))[0] for y in y_vt])

    x = x_v0[(x_v0 >= 15) & (x_v0 <= 130)]
    y = y_vt[(x_v0 >= 15) & (x_v0 <= 130)]

    xy = (np.take(x, x.size // 2) + 8, np.take(y, y.size // 2))
    offsetbox = TextArea(f"{np.round(t_diff, 2)}", textprops=dict(color="lightcoral", size=18, weight='bold'))
    ab = AnnotationBbox(offsetbox, xy,
                        bboxprops=dict(color='white',edgecolor='black',linewidth=2.0))
    axis.add_artist(ab)

    axis.plot(x, y, color = 'lightcoral', linewidth = 5.0)


def plot_tapg(v_0, t_apg, axis):

    x_min = 15.0
    x_max = v_0
    x_v0 = np.linspace(x_min, x_max, 100)
    y_vt = np.array([fsolve(vt_tapg_v0, 0.1, args=(t_apg, x))[0] for x in x_v0])

    y = y_vt[(y_vt >= 5.0) & (y_vt <= 65.0)]
    x = x_v0[(y_vt >= 5.0) & (y_vt <= 65.0)]

    #xy = ((max(x)+min(x))/2 - 10, (max(y)+min(y))/2)
    xy = (np.take(x, x.size // 2) + 6, np.take(y, y.size // 2)+2)
    offsetbox = TextArea(f"{np.round(t_apg, 2)}", textprops=dict(color="hotpink", size=18, weight='bold'))
    ab = AnnotationBbox(offsetbox, xy,
                        bboxprops=dict(color='white',edgecolor='black',linewidth=2.0))
    axis.add_artist(ab)

    axis.plot(x, y, color = "hotpink", linewidth = 5.0)


def plot_hapg(v_0, h_apg, axis):

    x_min = v_0
    x_max = 130.0
    x_v0 = np.linspace(x_min, x_max, 100)
    y_vt = np.array([fsolve(vt_hapg_v0, 0.1, args=(h_0, h_apg, x))[0] for x in x_v0])

    y = y_vt[(y_vt >= 5.0) & (y_vt <= 65.0)]
    x = x_v0[(y_vt >= 5.0) & (y_vt <= 65.0)]

    xy = (np.take(x, x.size // 2), np.take(y, y.size // 2)+3)
    offsetbox = TextArea(f"{np.round(h_apg, 2)}", textprops=dict(color="royalblue", size=18, weight='bold'))
    ab = AnnotationBbox(offsetbox, xy,
                        bboxprops=dict(color='white',edgecolor='black',linewidth=2.0))
    axis.add_artist(ab)

    axis.plot(x, y, color = 'royalblue', linewidth = 5.0)


def plot_corr_graph(axis):

    # Compute curves for the T_diff expression
    # NOTE: Putting this cell after the cells for H_apogee and T_apogee causes some error with Numba and JIT Compilation. 

    t_diffs = list(np.arange(0.2, 2.4, 0.2))
    t_apg_range = [(27, 65), (16, 65), (13.5, 65), (12.25, 65), (11, 65), (10, 65), (9.5, 65), (9, 65), (8, 65), (9.5, 65), (13, 65)]

    tdiff_plot_data = []

    for i, t_diff in enumerate(t_diffs):
        y_min, y_max = t_apg_range[i]
        y_vt = np.linspace(y_min, y_max, 100)
        x_v0 = [fsolve(v0_tdiff_vt, 0.1, args=(t_diff, y))[0] for y in y_vt]
        
        tdiff_plot_data.append((x_v0, y_vt))

    # Compute curves for the H_apogee expression

    x_mins = [18.629518238645137, 28.143139330117272, 35.78157652394566, 42.61606103657115, 49.031456808908565,
            55.22012137312889, 61.29626453630336, 67.33576876678198, 73.39338517921851, 79.51126001455683, 85.72359718436087,
            92.0594032856993, 98.54420083336248, 105.20115005394047, 112.05181397882923]
    h_apgs = list(range(20, 320, 20))

    h_plot_data = []

    for i, h_apg in enumerate(h_apgs):
        x_v0 = np.linspace(x_mins[i], 130.0, 100)
        y_vt = [fsolve(vt_hapg_v0, 0.1, args=(3, h_apg, x))[0] for x in x_v0]
        
        h_plot_data.append((x_v0, y_vt))

    # Compute curves for the t_apogee expression

    x_mins = [25, 25, 30, 35, 40, 46, 52, 59, 66, 73, 81, 91, 103]
    t_apgs = list(np.arange(2.0, 7.2, 0.4))

    t_plot_data = []

    for i, t_apg in enumerate(t_apgs):
        x_v0 = np.linspace(x_mins[i], 130.0, 100)
        y_vt = [fsolve(vt_tapg_v0, 0.1, args=(t_apg, x))[0] for x in x_v0]
        
        t_plot_data.append((x_v0, y_vt))

    h_labels = [f"{i}" for i in range(20, 320, 20)]
    h_labels.append("$h_{ap}$")

    t_labels = ["0.2 $\Delta t$", "0.4 $\Delta t$", "0.6 $\Delta t$", "0.8 $\Delta t$", "1.0 $\Delta t$", "1.2 $\Delta t$", 
           "1.4 $\Delta t$", "1.6 $\Delta t$", "1.8 $\Delta t$", "2.0 $\Delta t$", "2.2 $\Delta t$"]

    t_apg_labels = [f"{np.round(i, 2)}" for i in t_apgs]

    line1 = Line2D([0], [0], color='dodgerblue', linewidth=2)
    line2 = Line2D([0], [0], color='lightcoral', linewidth=2, linestyle='--')
    line3 = Line2D([0], [0], color='darkorchid', linewidth=2, linestyle='--')

    for i, (x, y) in enumerate(h_plot_data):
        axis.plot(x, y, color = 'dodgerblue', linewidth = 2.0)
        axis.text(x[-1]+2.5, y[-1], h_labels[i], fontsize = 12)
        if i == 14:
            axis.text(x[-1]+2.5, y[-1]+2.5, h_labels[i+1], fontsize = 14)
        
    for i, (x, y) in enumerate(t_plot_data):
        axis.plot(x, y, color = 'lightcoral', linestyle = '--', linewidth = 2.0)
        if i == 10 or i == 11:
            axis.text(x[0]-2, y[0]-1.5, t_apg_labels[i], fontsize = 12)
        else:
            axis.text(x[0]-2, y[0]+0.5, t_apg_labels[i], fontsize = 12)
        
    for i, (x, y) in enumerate(tdiff_plot_data):
        axis.plot(x, y, color = 'darkorchid', linestyle = '--', linewidth = 2.0)
        axis.text(x[-1], y[-1]+0.5, t_labels[i], fontsize = 13, rotation = 75)

    axis.set_title(r"Rocket Apogee Correlation", fontsize=18)
    axis.set_xlabel(r"Burnout Velocity $v_0$ (m/s)", fontsize=16)
    axis.set_ylabel(r"Terminal Velocity $v_t$ (m/s)", fontsize=16)
    axis.set_xlim(15, 140)
    axis.set_ylim(5, 70)
    #axis.grid()
    axis.legend([line1, line2, line3], ["$h_{ap} (m)$", "$t_{ap} (s)$", "$\Delta t (s)$"])
    axis.set_xticks(np.arange(20, 140, 10), size = 14)
    axis.set_yticks(np.arange(10, 70, 5), size = 14);

def print_results(result):
    v_0, v_t, v_end, t_0, t_apg, t_end, t_diff, h_apg = result
    print("BLAZE Correlation Results for Ballistic Rocket\n")
    print("Burnout Velocity:          {:10.3f}".format(v_0))
    print("Terminal Velocity:         {:10.3f}".format(v_t))
    print("Velocity at Touchdown:     {:10.3f}".format(v_end))
    print("Time to Burnout:           {:10.3f}".format(t_0))
    print("Time to Apogee:            {:10.3f}".format(t_apg))
    print("Time to Touchdown:         {:10.3f}".format(t_end))
    print("Time Difference:           {:10.3f}".format(t_diff))
    print("Apogee / Maximum Height:   {:10.3f}".format(h_apg))



def blaze(v_0 = None, v_t = None, h_apg = None, t_apg = None, t_diff = None):

    if (v_0 is not None) and (v_t is not None):

        v_end = V_END(v_0, v_t)
        h_apg = H_APG(v_0, v_t)
        t_0 = 2 * h_0/v_0
        t_apg = T_APG(v_0, v_t)
        t_tot = T_END(v_0, v_t)
        t_diff = t_tot - 2 * t_apg

    elif (v_0 is not None) and (h_apg is not None):

        while True:
            v_t = fsolve(vt_hapg_v0, np.random.randint(20, 60), args=(h_0, h_apg, v_0))[0]
            if (np.all(np.isclose(vt_hapg_v0(v_t, h_0, h_apg, v_0), [0.0])) and v_t > 0):
                break
        
        v_end = V_END(v_0, v_t)
        t_0 = 2 * h_0/v_0
        t_apg = T_APG(v_0, v_t)
        t_tot = T_END(v_0, v_t)
        t_diff = t_tot - 2 * t_apg
        
    elif (v_0 is not None) and (t_apg is not None):

        while True:
            v_t = fsolve(vt_tapg_v0, np.random.randint(20, 60), args=(t_apg, v_0))[0]
            if (np.all(np.isclose(vt_tapg_v0(v_t, t_apg, v_0), [0.0])) and v_t > 0):
                break
        
        v_end = V_END(v_0, v_t)
        h_apg = H_APG(v_0, v_t)
        t_0 = 2 * h_0/v_0
        t_tot = T_END(v_0, v_t)
        t_diff = t_tot - 2 * t_apg

    elif (v_0 is not None) and (t_diff is not None):

        vt_tdiff_v0 = lambda v_t_var : T_END(v_0, v_t_var) - 2 * T_APG(v_0, v_t_var) - t_diff

        while True:
            v_t = fsolve(vt_tdiff_v0, np.random.randint(20, 60))[0]
            if (np.all(np.isclose(vt_tdiff_v0(v_t), [0.0])) and v_t > 0):
                break

        v_end = V_END(v_0, v_t)
        h_apg = H_APG(v_0, v_t)
        t_0 = 2 * h_0/v_0
        t_apg = T_APG(v_0, v_t)
        t_tot = T_END(v_0, v_t)

    elif (v_t is not None) and (h_apg is not None):

        v0_hapg_vt = lambda v_0_var : h_0 - h_apg + ((v_t**2) / (2 * g)) * np.log(1 + ((v_0_var / v_t)**2))

        while True:
            v_0 = fsolve(v0_hapg_vt, np.random.randint(20, 60))[0]
            if (np.all(np.isclose(v0_hapg_vt(v_0), [0.0])) and v_0 > 0):
                break

        v_end = V_END(v_0, v_t)
        t_0 = 2 * h_0/v_0
        t_apg = T_APG(v_0, v_t)
        t_tot = T_END(v_0, v_t)
        t_diff = t_tot - 2 * t_apg

    elif (v_t is not None) and (t_apg is not None):

        v0_tapg_vt = lambda v_0_var: (2 * h_0) / v_0_var + (v_t / g) * np.arctan(v_0_var / v_t) - t_apg

        while True:
            v_0 = fsolve(v0_tapg_vt, np.random.randint(20, 60))[0]
            if (np.all(np.isclose(v0_tapg_vt(v_0), [0.0])) and v_0 > 0):
                break

        v_end = V_END(v_0, v_t)
        t_0 = 2 * h_0/v_0
        h_apg = H_APG(v_0, v_t)
        t_tot = T_END(v_0, v_t)
        t_diff = t_tot - 2 * t_apg

    elif (v_t is not None) and (t_diff is not None):

        v0_tdiff_vt = lambda v_0_var : T_END(v_0_var, v_t) - 2 * T_APG(v_0_var, v_t) - t_diff

        while True:
            v_0 = fsolve(v0_tdiff_vt, np.random.randint(20, 60))[0]
            if (np.all(np.isclose(v0_tdiff_vt(v_0), [0.0])) and v_0 > 0):
                break

        v_end = V_END(v_0, v_t)
        t_0 = 2 * h_0/v_0
        t_apg = T_APG(v_0, v_t)
        h_apg = H_APG(v_0, v_t)
        t_tot = T_END(v_0, v_t)
        print_results([v_0, v_t, v_end, t_0, t_apg, t_tot, t_diff, h_apg])


    elif (h_apg is not None) and (t_apg is not None):

        v_tap_hap = lambda v : [(2*h_0)/v[0] + (v[1]/g) * np.arctan(v[0] / v[1]) - t_apg, 
                                h_0 + (v[1]**2)/(2 * g) * np.log(1 + (v[0]/v[1])**2)-h_apg]
        
        while True:
            v_root = fsolve(v_tap_hap, np.random.randint(20, 60, 2))
            if (np.all(np.isclose(v_tap_hap(v_root), [0.0, 0.0])) and v_root[0] > 0 and v_root[1] > 0):
                break

        v_0, v_t = v_root
        v_end = V_END(v_0, v_t)
        t_0 = 2 * h_0/v_0
        t_tot = T_END(v_0, v_t)
        t_diff = t_tot - 2 * t_apg

    elif (h_apg is not None) and (t_diff is not None):

        v_hap_tdiff = lambda v : [T_END(v[0], v[1]) - 2 * T_APG(v[0], v[1]) - t_diff,
                                  h_0 + (v[1]**2)/(2 * g) * np.log(1 + (v[0] / v[1])**2)-h_apg]

        while True:
            v_root = fsolve(v_hap_tdiff, np.random.randint(20, 60, 2))
            if (np.all(np.isclose(v_hap_tdiff(v_root), [0.0, 0.0])) and v_root[0] > 0 and v_root[1] > 0):
                break

        v_0, v_t = v_root
        v_end = V_END(v_0, v_t)
        t_0 = 2 * h_0/v_0
        t_apg = T_APG(v_0, v_t)
        t_tot = T_END(v_0, v_t)

    elif (t_apg is not None) and (t_diff is not None):

        v_tap_tdiff = lambda v : [(2*h_0)/v[0] + (v[1]/g) * np.arctan(v[0] / v[1]) - t_apg, 
                                T_END(v[0], v[1]) - 2 * T_APG(v[0], v[1]) - t_diff]
        
        while True:
            v_root = fsolve(v_tap_tdiff, np.random.randint(20, 60, 2))
            if (np.all(np.isclose(v_tap_tdiff(v_root), [0.0, 0.0])) and v_root[0] > 0 and v_root[1] > 0):
                break

        v_0, v_t = v_root
        v_end = V_END(v_0, v_t)
        t_0 = 2 * h_0/v_0
        h_apg = H_APG(v_0, v_t)
        t_tot = T_END(v_0, v_t)
    
    else:
        print("Illegal Combination Entered")

    results = [v_0, v_t, v_end, t_0, t_apg, t_tot, t_diff, h_apg]
    print_results(results)
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    plot_corr_graph(axs)
    plot_tdiff(v_t, t_diff, axs)
    plot_tapg(v_0, t_apg, axs)
    plot_hapg(v_0, h_apg, axs)
    draw_v0_vt(v_0, v_t, axs)

    plt.savefig("Rocket_Apogee_Graph.pdf",bbox_inches='tight')

    return results