import plotly.tools as tls
import plotly.io as pio
import datetime
from matplotlib import cm
import matplotlib.colors as mcolors

import sys
from scipy.stats import mode
import networkx as nx
import springrank
from scipy.stats import ks_2samp, gaussian_kde
from scipy.interpolate import make_interp_spline
import pickle
import ast
import pandas as pd
import pprint
import numpy as np
import dask.dataframe as dd
from matplotlib.colors import Normalize

import seaborn as sns

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


def plot_connectivity_graph(g, inst):
    adj_matrix_sparse = nx.adjacency_matrix(g)

    adj_matrix_dense = adj_matrix_sparse.toarray()
    sr = springrank.SpringRank()
    sr.fit(adj_matrix_dense)
    ranks = sr.ranks
    sort_idx = np.flipud(np.argsort(ranks))
    a_sorted = adj_matrix_dense[:, sort_idx][sort_idx, :]
    g_reconstructed = nx.from_numpy_array(a_sorted)
    out_degree = np.sum(a_sorted, axis=1)

    in_degree = np.sum(a_sorted, axis=0)

    degree_difference = out_degree - in_degree
    data = np.where(degree_difference >= 0,
                    degree_difference / degree_difference.max(),
                    degree_difference / -degree_difference.min())

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')  # Set the figure background color
    ax.set_facecolor('black')  # Set the axes background color

    fig.set_dpi(180)
    colors = ['green' if val < 0 else 'gold' for val in data]
    ax.bar(range(len(data)), data, alpha=0.8, color=colors)
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(colors='white')

    ax.set_xlabel("Node ordered by rank", color='white')
    ax.set_ylabel('Normalized Out-degree - In-degree', color='white')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')

    plt.show()
    plt.savefig("figs/out_minus_in_{}.png".format(inst))


def plots(n_m, voltages, t_array, beta, k, inst):
    t = t_array
    V = voltages
    n_models = n_m

    sum_v = np.zeros_like(t)
    for j in range(n_models):
        v = [0 if x < 1 else 1 for x in V[j]]
        sum_v += np.array(v)
        plt.plot(t, np.array(v)+j, label=f'ff {j + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential')
    plt.title('Multiple Interacting Two Timescale Integrate-and-Fire Models, beta={}'.format(beta))
    plt.legend()

    plotly_fig = tls.mpl_to_plotly(plt.gcf())

    pio.write_html(plotly_fig, file='figs/two_ts_figure_b{}_k{}_{}.html'.format(beta, k, inst), auto_open=True)
    print('figs/all_two_ts_figure_b{}_k{}_{} written'.format(beta, k, inst))
    plt.clf()

    plt.plot(t, sum_v, label='Summed Voltages', color='black', linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Number of flashes')
    plt.title('Sum of Multiple Interacting Two Timescale Integrate-and-Fire Models, beta={}'.format(beta))
    plt.legend()

    plotly_fig = tls.mpl_to_plotly(plt.gcf())

    pio.write_html(plotly_fig, file='figs/sum_two_ts_figure_b{}_k{}_{}.html'.format(beta, k, inst), auto_open=True)
    print('figs/sum_two_ts_figure_b{}_k{}_{} written'.format(beta, k, inst))


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


def load_parameter_sweep_data(p):
    # Define a function to concatenate lists
    def concatenate_lists(series):
        return [item for sublist in series for item in ast.literal_eval(sublist)]

    fields = ['beta', 'k_thresh', 'driven_freq', 'interflashes_1', 'interbursts_1', 'spiketimes_1']

    with open(p, 'r') as f:
        res = pd.read_csv(f, usecols=fields)

    print('dataframe loaded from {} at {}'.format(p.split('/')[1][:-4], datetime.datetime.now().time()))

    res_combined = res.groupby(['beta', 'driven_freq', 'k_thresh'])['interflashes_1'].apply(
        concatenate_lists).reset_index()

    print('grouped at {}'.format(datetime.datetime.now().time()))

    print('{} with concatenated interflashes combined and returned at {}'.format(p.split('/')[1][:-4],
                                                                                 datetime.datetime.now().time()))
    return res_combined


def get_color_from_mesh(beta):
    cmap = plt.get_cmap('copper_r')

    return cmap(beta)


def compare_statistic(l1, l2, comp='ks'):
    if comp == 'ks':
        return ks_2samp(l1, l2)[0]
    elif comp == 'median':
        l1 = [x for x in l1 if not np.isnan(x)]
        l2 = [x for x in l2 if not np.isnan(x)]

        return abs(np.median(l1) - np.median(l2))
    elif comp == 'mode':
        l1 = np.array([x for x in l1 if not np.isnan(x)])
        l2 = np.array([x for x in l2 if not np.isnan(x)])
        kde_a = gaussian_kde(l1)
        kde_b = gaussian_kde(l2)
        x_vals = np.linspace(min(l1.min(), l2.min()), max(l1.max(), l2.max()), 1000)
        mode_a = x_vals[np.argmax(kde_a(x_vals))]
        mode_b = x_vals[np.argmax(kde_b(x_vals))]

        return abs(mode_a - mode_b)
    else:
        # Approximate bayesian computation here

        return abs(np.mean(l1) - np.mean(l2))


def map_value_in_range(val, in_min=0, in_max=1, out_min=0.2, out_max=0.8):
    val = max(in_min, min(val, in_max))

    return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)


def compare_models(experimental, list_of_sims, plot=False):
    colormap = cm.get_cmap('Spectral', 24)
    comparison_across = {}

    for compare_method in ['mode']:  # ['ks','median', 'mode'], 'abc' pending
        freq_level = {}
        model_min_sums = []
        model_level_heatmaps = []
        min_stat_all = 999
        max_stat_all = -1
        for s in list_of_sims:
            freq_level[s[0]] = {}
            model_min_sum = 0
            df_combined = s[1]
            betas = df_combined['beta'].unique()
            ks = df_combined['k_thresh'].unique()
            xs = df_combined['driven_freq'].unique()

            # x is an instance of a simulated distribution
            for ii, x in enumerate(xs):
                if ii == 4:
                    color = 'yellow'
                else:
                    color = colormap.__call__(ii * 3)
                if plot:
                    bigfig, bigax = plt.subplots()
                else:
                    bigfig = None
                    bigax = None
                experimental_x = experimental[x]

                # Accounting: get max and min statistics
                min_line_index = -1
                min_stat = 999
                index = -1
                min_vals = None
                max_line_index = -1
                max_stat = 0
                max_vals = None
                for beta in betas:
                    for k in ks:
                        index += 1
                        if plot:
                            fig1, ax1 = plt.subplots()
                        else:
                            fig1 = None
                            ax1 = None
                        subset = df_combined[(df_combined['beta'] == beta) & (df_combined['k_thresh'] == k) & (
                                    df_combined['driven_freq'] == x)]['interflashes_1']

                        # subset is the simulated data at the particular beta and k required
                        # experimental[x] is the real data comparator
                        # bigfig bigax has all the subsets plotted against the experimental, with the best fit
                        # made bold and prominent
                        # once again, best fit is KS but should be ABC

                        y, bin_edges = np.histogram(np.array(subset.values[0]), bins=np.arange(0.0, 2.0, 0.04), density=True)
                        ys = [height for height in y]
                        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                        x_nice = np.linspace(min(bin_centers), max(bin_centers), 300)
                        _nice = make_interp_spline(bin_centers, ys)
                        y_nice = _nice(x_nice)
                        y_nice = [y if y > 0.0 else 0.0 for y in y_nice]

                        c_statistic = compare_statistic(subset.iloc[0], experimental[x], compare_method)
                        if c_statistic < min_stat:
                            min_stat = c_statistic
                            min_line_index = index
                            min_vals = (beta, k)
                            if min_stat < min_stat_all:
                                min_stat_all = min_stat
                        elif c_statistic > max_stat:
                            max_stat = c_statistic
                            max_line_index = index
                            max_vals = (beta, k)
                            if max_stat > max_stat_all:
                                max_stat_all = max_stat
                        if freq_level[s[0]].get(x) is None:
                            freq_level[s[0]][x] = [
                                (beta, k, c_statistic)]
                        else:
                            freq_level[s[0]][x].append(
                                (beta, k, c_statistic))
                        individual_sim_color = get_color_from_mesh(beta)
                        if plot:
                            bigax.plot(x_nice, y_nice, color=individual_sim_color, alpha=map_value_in_range(k), lw=1)
                            ax1.hist(subset, density=True, bins=np.arange(0.0, 2.0, 0.04), color='grey', alpha=0.7)
                            ax1.hist(experimental_x, density=True, bins=np.arange(0.0, 2.0, 0.04),
                                     color=color, alpha=0.7)
                        if plot:
                            if compare_method == 'mode':
                                plt.savefig('figs/histogram_comparisons/{}{}_{}ms_{}beta_{}k_{}.png'.format(
                                    c_statistic, compare_method, x, beta, k, s[0])
                                )
                            plt.close(fig1)
                if plot:
                    bigax.lines[min_line_index].set_linewidth(4)
                    bigax.lines[min_line_index].set_zorder(3)
                    bigax.lines[min_line_index].set_alpha(1.0)
                    if min_vals:
                        label = '{}beta {}k'.format(min_vals[0], min_vals[1])
                    else:
                        label = 'NAbeta NAk'
                    bigax.lines[min_line_index].set_label('Best performing simulation, {}'.format(label))
                    plt.draw()
                expy, expbin_edges = np.histogram(np.array(experimental_x),
                                                  bins=np.arange(0.0, 2.0, 0.04), density=True)
                expys = [height for height in expy]
                expbin_centers = 0.5 * (expbin_edges[1:] + expbin_edges[:-1])
                expx_nice = np.linspace(min(expbin_centers), max(expbin_centers), 300)
                exp_nice = make_interp_spline(expbin_centers, expys)
                expy_nice = exp_nice(expx_nice)
                expy_nice = [y if y > 0.0 else 0.0 for y in expy_nice]
                if plot:
                    cmap = cm.get_cmap('copper_r')
                    norm = mcolors.Normalize(vmin=0, vmax=1)
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = bigfig.colorbar(sm, orientation='vertical', ax=bigax)
                    cbar.set_label('Beta')

                    bigax.plot(expx_nice, expy_nice, color=color, lw=4, zorder=5, label='Exp. results')
                    plt.legend()
                    plt.savefig('figs/all_stacked_{}_{}_{}comp.png'.format(x, s[0], compare_method))
                    plt.close(bigfig)
            to_heatmap = {}
            to_heatmap_sums = []
            for i, beta in enumerate(betas):
                for j, k in enumerate(ks):
                    sum_of_vals = 0
                    for df_i, driven_freq in enumerate(xs):
                        subset = df_combined[(df_combined['beta'] == beta) & (df_combined['k_thresh'] == k) & (
                                df_combined['driven_freq'] == driven_freq)]['interflashes_1']
                        val = (beta, k, compare_statistic(subset.iloc[0], experimental[driven_freq], compare_method))
                        sum_of_vals += val[2]
                        if to_heatmap.get(driven_freq) is None:
                            to_heatmap[driven_freq] = [val]
                        else:
                            to_heatmap[driven_freq].append(val)
                    to_heatmap_sums.append((beta, k, sum_of_vals))
            df_sums = pd.DataFrame(to_heatmap_sums, columns=['beta', 'k', 'sum_ks_stat'])
            min_row = df_sums.loc[df_sums['sum_ks_stat'].idxmin()]
            min_beta = min_row['beta']
            min_k = min_row['k']
            min_ks_stat = min_row['sum_ks_stat']
            print('Model: {}, min_beta: {}, min_k: {}, min_{}_stat: {} summed across freqs'.format(
                s[0],
                min_beta,
                min_k,
                compare_method,
                min_ks_stat)
            )
            all_min_stats = []
            model_level_heatmap = df_sums.pivot('beta', 'k', 'sum_ks_stat')
            model_level_heatmaps.append(model_level_heatmap)
            if compare_method == 'ks' or compare_method == 'median':
                norm_all = Normalize(vmin=np.min(min_stat_all), vmax=np.max(max_stat_all))
            else:
                norm_all = Normalize(vmin=np.min(0.22), vmax=np.max(1.4))

            ###
            # Figure generating heatmap of all beta and k values for each simulation type summed across all frequencies
            ###

            if plot:
                plt.figure(figsize=(10, 6))
                sns.heatmap(model_level_heatmap, annot=True, fmt=".4f", cmap="YlGnBu", norm=norm_all)
                plt.title('{} diff for {} model between exp. dist and sim dist'.format(compare_method, s[0]))
                plt.xlabel('k')
                plt.ylabel('beta')
                plt.savefig('figs/Heatmap_all_freqs_model_{}_method_{}.png'.format(s[0], compare_method))
                plt.close()
            for k in to_heatmap.keys():
                df = pd.DataFrame(to_heatmap[k], columns=['beta', 'k', 'ks_stat'])
                min_row = df.loc[df['ks_stat'].idxmin()]
                min_beta = min_row['beta']
                min_k = min_row['k']
                min_ks_stat = min_row['ks_stat']
                model_min_sum += min_ks_stat
                all_min_stats.append(min_ks_stat)
                to_heatmap_sums.append((min_beta, min_k, min_ks_stat))
                heatmap_data = df.pivot('beta', 'k', 'ks_stat')
                if compare_method == 'mode':
                    if comparison_across.get(s[0]):
                        comparison_across[s[0]].append((k, min_ks_stat))
                    else:
                        comparison_across[s[0]] = [(k, min_ks_stat)]
                print('Model: {}, LED_Freq: {}, min_beta: {}, min_k: {}, {}: {}'.format(
                    s[0], k, min_beta, min_k, compare_method, min_ks_stat))

                ###
                # Figure making heatmap for each frequency separately
                ###

                if plot:
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu")
                    plt.title('{} diff for {} model between exp. dist and sim dist, driven_freq = {}'.format(
                        compare_method, s[0], k))
                    plt.xlabel('k')
                    plt.ylabel('beta')
                    plt.savefig('figs/Heatmap_{}_diff_driven_freq_{}_model_{}.png'.format(compare_method, k, s[0]))
                    plt.close()

            model_min_std = np.std(all_min_stats)
            model_min_mean = np.mean(all_min_stats)

            model_min_sums.append((s[0], model_min_sum, model_min_std, model_min_mean))
        combined_df = pd.concat(model_level_heatmaps, ignore_index=True)
        min_combined = min(combined_df.values.flatten())
        max_combined = max(combined_df.values.flatten())

        ###
        # Figure attempting to compare all the beta and k value statistics in a plane
        ###

        dfs_to_plot = []
        for model, df in zip(['Excitatory', 'Excitatory-refractory', 'Excitatory-inhibitory'], model_level_heatmaps):
            dfs_to_plot.append((model, df))
        if plot:
            fig = plt.figure(figsize=(18, 6))

            colors = ['cividis', 'cividis', 'cividis']  # Colormaps for each DataFrame
            elev = 20  # Elevation angle
            azim = -60  # Azimuth angle
            for i, (model, df) in enumerate(dfs_to_plot):
                ax = fig.add_subplot(1, 3, i + 1, projection='3d')

                rows, cols = df.shape
                x = np.arange(rows)
                y = np.arange(cols)
                x, y = np.meshgrid(x, y)
                z = df.values.T  # Transpose to align with x and y
                norm = Normalize(vmin=np.min(z), vmax=np.max(z))

                ax.plot_surface(x, y, z, cmap=colors[i], norm=norm, alpha=0.7)

                ax.set_title(model)
                ax.set_xlabel('Beta')
                ax.set_ylabel('K')
                ax.set_zlabel('Normalized Sum {} Difference'.format(compare_method))
                ax.set_zlim(min_combined - 0.05, max_combined + 0.05)
                ax.view_init(elev=elev, azim=azim)  # Set consistent view angle

            plt.tight_layout()
            plt.savefig('figs/Normalized_Sum_{}_Difference_plane_{}.png'.format(compare_method, s[0]))
            plt.close()

        for (m,kss,mms,mmm) in model_min_sums:
            print('Model: {}, min_{}_stat_sum: {}, min_{}_stat_std: {}, min_{}_stat_mean: {}'.format(
                m, compare_method, kss, compare_method, mms, compare_method, mmm)
            )

    ###
    # Figure comparing best results from each simulation type
    ###
    print(comparison_across)
    if plot:
        colors = {'Excitatory': 'gray',
                  'Excitatory-inhibitory': 'limegreen',
                  'Excitatory-refractory': 'rosybrown'}
        fig, ax = plt.subplots()
        for i,k in enumerate(comparison_across.keys()):
            std = 0
            for result in model_min_sums:
                if k in result:
                    std = result[3]
            xs = [x[0] for x in comparison_across[k]]
            ys = [x[1] for x in comparison_across[k]]
            ax.errorbar(xs, ys, yerr=(std / 2), color=colors[k], capsize=2, label=k)
            ax.scatter(xs, ys, color=colors[k])
        ax.set_xlabel('LED Freq(s)')
        ax.set_ylabel('Min KS stat between experimental result and simulated result')
        ax.set_title('Model comparison')
        plt.legend()
        plt.savefig('figs/Aggregate_model_comparison.png')
        plt.close()

    print('Finished comparing!')
    if plot:
        print('plots generated')
    else:
        print('plots not generated, pass --plot to generate plots')
