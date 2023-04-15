import numpy as np
import matplotlib.pyplot as plt

def compute_velocity(t_0, t_ap, t_end, v_t):
    # Define the gravitational constant
    g = 9.81

    # Create time arrays for the first and second segments of the curve
    t1 = np.linspace(t_0, t_ap, 500)
    t2 = np.linspace(t_ap, t_end, 500)

    # Initialize velocity array with the initial velocity (0.0)
    velocities = [0.0]

    # Calculate velocities for the first segment (t1)
    for t in t1:
        v1 = v_t * np.tan(g * (t_ap - t) / v_t)
        velocities.append(v1)

    # Calculate velocities for the second segment (t2)
    for t in t2:
        v2 = v_t * np.tanh(g * (t_ap - t) / v_t)
        velocities.append(v2)

    # Combine time arrays, including the initial time (0.0) and a small increment past t_end
    times = [0.0] + list(t1) + list(t2) + [t_end + 0.01]

    # Add the final velocity (0.0) to the velocities array
    velocities.append(0.0)

    return times, velocities


def compute_height(t_0, t_ap, t_end, v_t, h_ap):
    # Define the gravitational constant
    g = 9.81

    # Create time arrays for the first and second segments of the curve
    t1 = np.linspace(t_0, t_ap, 500)
    t2 = np.linspace(t_ap, t_end, 500)

    # Initialize height array with the initial height (0.0)
    heights = [0.0]

    # Calculate heights for the first segment (t1)
    for t in t1:
        h1 = h_ap + (pow(v_t, 2) / g) * np.log(np.cos(g * (t_ap - t) / v_t))
        heights.append(h1)

    # Calculate heights for the second segment (t2)
    for t in t2:
        h2 = h_ap - (pow(v_t, 2) / g) * np.log(np.cosh(g * (t_ap - t) / v_t))
        heights.append(h2)

    # Combine time arrays, including the initial time (0.0)
    times = [0.0] + list(t1) + list(t2)

    return times, heights

def plot_vt_ht(t_0, t_ap, t_end, v_t, h_ap):
    t1, v = compute_velocity(t_0, t_ap, t_end, v_t)
    t2, h = compute_height(t_0, t_ap, t_end, v_t, h_ap)

    fig = plt.figure(figsize=(14,5))

    ax1 = fig.add_subplot(121)
    ax1.plot(t1, v, linewidth=3)
    ax1.grid()
    ax1.set_xlabel("Time (s)", size=16)
    ax1.set_ylabel("Velocity (m/s)", size=16)

    ax2 = fig.add_subplot(122)
    ax2.plot(t2, h, linewidth=3)
    ax2.grid()
    ax2.set_xlabel("Time (s)", size=16)
    ax2.set_ylabel("Altitude (m)", size=16)
    plt.savefig("Velocity_Height_Graph.pdf",bbox_inches='tight');