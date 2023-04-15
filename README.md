
# BLAZE: Ballistic Launch Analyser and Zenith Estimator
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vijeycreative/BLAZE/HEAD?labpath=Example.ipynb)

BLAZE (Ballistic Launch Analyzer and Zenith Estimator) is an easy-to-use Python program designed to calculate apogee, thrust, and other essential metrics for vertically launched rockets using minimal data input. When launching rockets, determining their peak altitude is crucial. Currently, there are two methods: (1) Attach a costly altimeter to the rocket and hope it survives potential crashes, or (2) utilize trigonometric data from [one](http://waterrocket.uh-lab.de/seamcalc.htm) or more distant observers equipped with inclinometers and compasses. BLAZE introduces a third approach, developed by Dean Wheeler, which only needs a single remote observer with a stopwatch.

This innovative method for determining apogee relies on a mathematical model that describes rocket flight after the thrust phase. Although the model's development involves calculus-based classical physics, users don't need to be familiar with this to benefit from the model's outcomes. Dean Wheeler offers a [graphical technique](https://www.et.byu.edu/~wheeler/benchtop/flight.php#graphical) and a [simple equation](https://www.et.byu.edu/~wheeler/benchtop/flight.php#simplest) for those with less mathematical expertise to compute rocket apogee heights. However, the BLAZE program effortlessly calculates these values without the need for graphical methods.

It's important to note that the findings presented here are also applicable to ballistic objects other than water rockets.

## Computing with BLAZE

The BLAZE program (also this [graph](https://www.et.byu.edu/~wheeler/benchtop/pix/apogee.pdf)) establishes a relationship between five key metrics:

1.  Terminal Velocity, $v_t$ - For ballistic flights, this value is consistent throughout the entire flight. For recovery-deployment flights, it only applies during the ascent phase. Regardless, this value should remain constant from one flight to another, meaning it only needs to be determined once for a specific rocket.
2.  Burnout Velocity, $v_0$ - This variable may change between flights unless the water quantity and pressure are carefully controlled and set to specific values.
3.  Apogee Height, $h_{ap}$ - This can be independently measured using an altitude sensor inside the rocket or an inclinometer or theodolite.
4.  Time to Apogee, $t_{ap}$ - This value should be obtainable for every rocket flight. (Note that it is easier to identify the apogee point when standing at a reasonable distance from the launcher.)
5.  Time Difference, $\Delta t = t_{end} - 2t_{ap}$ - This value can be collected _only_ during a ballistic flight. Both $t_{ap}$ and $t_{end}$ can be gathered during the flight using a stopwatch with a multi-lap storage feature or by recording the flight with a video camera for later analysis.

**The crucial point is that knowing just two of the five metrics allows you to calculate the remaining three.** This implies that meticulously documented flight times for a ballistic flight (without a recovery mechanism) can provide the terminal velocity, burnout velocity, and apogee height.

## Assumptions

In this technique, two primary assumptions are made. Let's take a closer look at them:

1.  We assumed that the drag coefficient remains constant. In reality, it somewhat depends on velocity—usually, $C_D$ decreases as velocity increases. This means that the terminal velocity and thus $C_D$ obtained through this method will be an average value across various velocities. As a result, the calculated apogee heights and burnout velocities based on time measurements might be overestimated by a small percentage.
2.  We assumed that the flight is strictly vertical. However, no flight is purely vertical due to wind currents and non-vertical launch angles. Nevertheless, the error in a calculated apogee is less than a couple of percent as long as the launch angle stays within 10 degrees of vertical. This method yields satisfactory results without the need for meticulous adjustments for wind and nonvertical trajectories, which are necessary in trigonometric methods.

Since the BLAZE program is designed for water bottle rockets, we make reasonable assumptions about the burnout height $h_0 = 3$ m and time to burnout $t_0 = \frac{2h_0}{v_0}$. This approximation is valid because water rockets are characterized by their rapid thrust phase. Typically, it takes less than $0.1$ seconds to expel all the water, followed by an additional $0.05$ seconds to release the remaining high-pressure air. Once the thrust phase is complete, a certain amount of time will have passed since lift off, and the rocket will be at a specific height above the ground with a particular velocity.

## Usage

To utilize the BLAZE program, it's not necessary to download the scripts, though you have the option to do so. You can access the BLAZE program directly in your web browser by clicking the Binder badge at the top of this readme file or by following this [link](https://mybinder.org/v2/gh/vijeycreative/BLAZE/HEAD?labpath=Example.ipynb). Both the Binder badge and the provided link will launch the Example.ipynb notebook on Jupyter Lab, hosted on Binder's server. The second cell of the Jupyter notebook contains a function call to the `blaze(v_0 = None, v_t = None, h_apg = None, t_apg = 3.24, t_diff = 1.48)` program, with only $t_{ap}$ and $\Delta t$ values entered in this instance. To use the program with any two other quantities, simply input their values in the corresponding variables and set all other variables to `None`.

## Acknowledgements

When utilizing this program for school projects or university assignments, please consider acknowledging Dean Wheeler's webpage [Dean's Benchtop](https://www.et.byu.edu/~wheeler/benchtop/) as well as V Vijendran and this repository in your citations.

## Intellectual Property

The concepts and methods employed by the BLAZE program are the intellectual property of Dean Wheeler. Unless explicitly authorized in advance, you are prohibited from storing, distributing, modifying, or reproducing any part of the content beyond what is permitted under the "[fair use](http://www.copyright.gov/fls/fl102.html)" provision of U.S. copyright law. If the intended use is non-commercial and educational, it is likely covered by this provision.
