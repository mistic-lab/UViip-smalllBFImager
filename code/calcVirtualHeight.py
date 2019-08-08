import numpy as np
import math
import DRAO_utils as du
import matplotlib.pyplot as plt


def get_phase_diff_as_dist(phase_diff_rad, frequency):
    """
    Return distance in m

    Parameters
    ----------
    phase_diff_rad : float
        difference in receiving path length in rad
    frequency : int
        frequency to calculate it at in Hz (optional)
    """
    if 'frequency' not in locals():
        frequency = du.get_frequency()*1000
    c = 3e8
    wavelength = c/(frequency)
    phase_diff_m = wavelength*phase_diff_rad/math.pi
    return float(phase_diff_m)


def find_virtual_height_triangles(phase_diff_m, x1, x2):
    """
    Return height of ionosphere

    Parameters
    ----------
    phase_diff_m : float
        difference in receiving path length in m
    x1 : float
        distance from transmitter to first antenna
    x2 : float
        distance from first antenna to second antenna
    """
    phase_diff_m = phase_diff_m/2  # Half wave limit (fx correlator causes pi to mean pi/2)

    x2 = x2+x1

    h = math.sqrt((((x1**2)/4 + phase_diff_m**2 + (x2**2)/4)/(-2*phase_diff_m))**2 - (x1**2)/4)

    return h


def find_virtual_height_plane_wave(phase_diff_m, x1, x2):
    """
    Return height of ionosphere

    Parameters
    ----------
    phase_diff_m : float
        difference in receiving path length in m
    x1 : float
        distance from transmitter to first antenna
    x2 : float
        distance from first antenna to second antenna
    """

    phase_diff_m = phase_diff_m/2  # Half wave limit (fx correlator causes pi to mean pi/2)

    x3 = x1+x2  # Distance from TX to second antenna

    theta = math.acos(phase_diff_m/x2)
    # print("angle of arrival: {}".format(math.degrees(theta)))
    low = math.tan(theta)*x1/2
    high = math.tan(theta)*x3/2
    avg = abs(high-low)/2 + min(high, low)
    return avg


def plot_height_vs_phase(freq, dist_between_ants):
    """
    Simulate height vs phase

    Parameters
    ----------
    freq : int
        frequency to calc at in Hz
    dist_between_ants : float
        distance between receiver ants in m
    """
    highs = np.zeros(100)
    hs = np.zeros(99)

    phases = np.linspace(0, 3.14/2, 100)

    for i in range(0, 100):
        # print(phases[i])
        phase_dist = get_phase_diff_as_dist(phases[i], freq)
        highs[i] = find_virtual_height_plane_wave(phase_dist, 19840, dist_between_ants)
        if i > 0:
            hs[i-1] = find_virtual_height_triangles(phase_dist, 19840, 19840+dist_between_ants)
    highs = highs[1:]/1000
    phases = phases[1:]
    hs = hs/1000

    plt.figure(0)
    plt.plot(phases, highs)
    plt.title('Plane-wave approx. at {}MHz, {}m between antennas'.format(freq/1e6, dist_between_ants))
    plt.xlabel('Phase diff (rad)')
    plt.ylabel('Ionospheric height (km)')
    plt.show(block=False)

    plt.figure(1)
    plt.plot(phases, hs)
    plt.title('Triangle approx. at {}MHz, {}m between antennas'.format(freq/1e6, dist_between_ants))
    plt.xlabel('Phase diff (rad)')
    plt.ylabel('Ionospheric height (km)')
    plt.show(block=False)

    plt.figure(2)
    plt.plot(phases, highs)
    plt.yscale('log')
    plt.title('Plane-wave approx. at {}MHz, {}m between antennas'.format(freq/1e6, dist_between_ants))
    plt.xlabel('Phase diff (rad)')
    plt.ylabel('Ionospheric height (km)')
    plt.show(block=False)

    plt.figure(3)
    plt.plot(phases, hs)
    plt.yscale('log')
    plt.title('Triangle approx. at {}MHz, {}m between antennas'.format(freq/1e6, dist_between_ants))
    plt.xlabel('Phase diff (rad)')
    plt.ylabel('Ionospheric height (km)')
    plt.show(block=False)
