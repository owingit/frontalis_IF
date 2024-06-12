import plotly.tools as tls
import plotly.io as pio
from scipy.stats import mode
import pickle
import pandas as pd
import pprint
import numpy as np

import matplotlib.pyplot as plt


def cost_function(x, y, max_x, max_y, w1=1, w2=2.0):
    # Normalize x and y
    normalized_x = x / max_x
    normalized_y = y / max_y
    if normalized_x == 0:
        penalty_x = float('inf')
    else:
        penalty_x = 1 / normalized_x
    penalty_y = normalized_y
    return w1 * penalty_x + w2 * penalty_y


def calculate_intervals(ts):
    n = len(ts)

    indices = []

    for i in range(n):
        if ts[i] == 1:
            indices.append(i)

    intervals = []
    if len(indices) > 1:
        for j in range(1, len(indices)):
            intervals.append(indices[j] - indices[j - 1])

    return intervals


def calculate_percentage(ts_a, ts_b, window_size):
    if len(ts_a) != len(ts_b):
        raise ValueError("Time series must have the same length")

    n = len(ts_a)
    total_numerator = 0
    total_denominator = 0

    # Sliding window of size 90 (3s)
    # only add to the denominator if
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        numerator = 0
        denominator = 0
        denominator_a_component = 0
        denominator_b_component = 0

        for i in range(start, end):
            window_a = ts_a[max(start, i-1):min(n, i+2)]
            window_b = ts_b[max(start, i-1):min(n, i+2)]

            # Check if there is at least one '1' in both windows
            if 1 in window_a and 1 in window_b:
                numerator += 1

            # Count the total number of '1's in both windows
            count_ones_a = window_a.count(1)
            count_ones_b = window_b.count(1)
            denominator_a_component += count_ones_a
            denominator_b_component += count_ones_b

            denominator += count_ones_a + count_ones_b

        if denominator_b_component > 0 and denominator_a_component > 0:
            total_numerator += numerator
            total_denominator += denominator

    if total_denominator == 0:
        percentage = 0  # To handle the case when the denominator is zero
    else:
        percentage = (total_numerator / total_denominator) * 100

    return percentage


def plots(n_m, voltages, t_array, beta):
    t = t_array
    V = voltages
    n_models = n_m
    # Plot the results
    for j in range(n_models):
        v = [0 if x < 1 else 1 for x in V[j]]
        plt.plot(t, np.array(v)+j, label=f'ff {j + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential')
    plt.title('Multiple Interacting Two Timescale Integrate-and-Fire Models, beta={}'.format(beta))
    plt.legend()

    # Convert the Matplotlib figure to a Plotly figure
    plotly_fig = tls.mpl_to_plotly(plt.gcf())

    # Save the Plotly figure as an HTML file
    pio.write_html(plotly_fig, file='figs/two_ts_figure_{}.html'.format(beta), auto_open=True)


def vis(stats, stat_keys, max_percentage, max_difference):
    for beta in stats.keys():
        print('Interflash mode difference: ', np.mean(stats[beta]['interflash_mode_difference']), 'Beta:', beta)
        print('Sync percent: ', np.mean(stats[beta]['percentage']), 'Beta: ', beta)
        print('Cost function: ', cost_function(
            np.mean(stats[beta]['percentage']),
            np.mean(stats[beta]['interflash_mode_difference']),
            max_percentage, max_difference),
              'Beta: ', beta)

    for stat_key in stat_keys:
        fig, ax = plt.subplots()
        for beta in stats.keys():
            if stat_key != 'percentage':
                ax.hist(stats[beta][stat_key], density=True, bins=np.arange(0.0, 10.0, 0.1), label=beta)
            else:
                ax.hist(stats[beta][stat_key], density=True, bins=np.arange(0.0, 100.0, 1.0), label=beta)
        plt.legend()
        ax.set_title(stat_key)
        plt.show()

    pp = pprint.PrettyPrinter()
    pp.pprint(stats)
