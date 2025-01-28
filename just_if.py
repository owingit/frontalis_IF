import numpy as np
import math
import random
import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Manager, Process, Pool

import argparse

import analysis


class Simulation:
    def __init__(self, trial_num, trial_beta, trial_k, if_df, fc_df, ib_df, trial_td, trial_driven, trial_driver,
                 trial_driven_freq, trial_n_models, trial_T, dt, model_specifics, trial_log, attn, on_grid, scale_same):
        self.trial_num = trial_num
        self.beta = trial_beta
        self.k = trial_k
        self.if_df = if_df
        self.fc_df = fc_df
        self.ib_df = ib_df
        self.td = trial_td
        self.driven = trial_driven
        self.driver = trial_driver
        self.driven_freq = trial_driven_freq
        self.n_models = trial_n_models
        self.T = trial_T
        self.dt = dt
        self.model = model_specifics
        self.log = trial_log
        self.attn = attn
        self.stats = None
        self.v_results = None
        self.status = False
        self.time_vector = None
        self.scale_same = scale_same
        random_walks = np.random.normal(0, np.sqrt(self.dt), (int(self.T/self.dt), self.n_models))

        w_p = np.cumsum(random_walks, axis=0)
        self.w_p = (w_p - np.min(w_p, axis=0)) / (np.max(w_p, axis=0) - np.min(w_p, axis=0))

        self.attns = np.full(self.n_models, self.attn)
        self.precomputed_distances = {}
        self.default_rate_scale = 5

        if on_grid:
            self.connectivity_graph, self.node_positions = self.initialize_grid_network()
        else:
            self.connectivity_graph = None
            self.node_positions = None

    def initialize_grid_network(self):
        size = int(math.sqrt(self.n_models))
        spacing = 15
        grid_length = size * spacing

        G = nx.grid_2d_graph(size, size)

        for (u, v) in list(G.edges()):
            if random.random() >= 0.0:
                G.remove_edge(u, v)

        positions = {}
        for x in range(size):
            for y in range(size):
                node = (x, y)
                positions[node] = (x * spacing, y * spacing)

        nx.set_node_attributes(G, positions, 'pos')

        return G, positions

    def compute_distance(self, node1, node2):
        # Get the positions of the two nodes
        x1, y1 = self.node_positions[node1]
        x2, y2 = self.node_positions[node2]

        if self.precomputed_distances.get((node1, node2)) is None:
            # Calculate Euclidean distance
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            self.precomputed_distances[(node1, node2)] = distance
        else:
            distance = self.precomputed_distances[(node1, node2)]
        return distance

    def map_to_2d(self, index):
        r = int(math.sqrt(self.n_models))
        if not (0 <= index < self.n_models):
            raise ValueError("Index must be in the range 0 to {}.".format(self.n_models - 1))
        y = index // r
        x = index % r
        return x, y

    def simulate(self):
        ##
        #
        # Simulates the two-timescale light-controlled oscillator dynamics with n_models individuals
        # The model_selection 'E' runs excitatory-only dynamics (beta is always positive)
        # The model_selection 'ER' runs excitatory-refractory dynamics (beta is always positive or 0)
        # The model_selection 'EI' runs excitatory-inhibitory dynamics (beta is positive or negative)
        # Simulation dynamics are carried out on a (n_models, timesteps) array
        #
        ##

        ks = np.zeros(self.n_models)
        ks[:] = self.k
        b = self.beta
        ms = self.model
        print('Trial {} - k {} - beta {} - driven freq {}'.format(self.trial_num,
                                                                  self.k,
                                                                  self.beta,
                                                                  self.driven_freq))
        t = np.arange(0, self.T, self.dt)  # Time vector
        initial_tcs = self.if_df.sample(self.n_models, ignore_index=True).to_numpy().reshape(1, self.n_models)
        if self.scale_same:
            min_val, max_val = 0.58, 0.63
            original_min, original_max = initial_tcs.min(), initial_tcs.max()
            scaled_tcs = ((initial_tcs - original_min) / (original_max - original_min)) * (max_val - min_val) + min_val
            initial_tcs = scaled_tcs
        initial_fcs = self.fc_df.sample(self.n_models, ignore_index=True).to_numpy().reshape(1, self.n_models)
        if self.driven:
            initial_tcs[0][self.driver] = self.driven_freq
            initial_fcs[0][self.driver] = len(t) - 1  # never runs out
            initial_sleep_period = 60 / self.dt
        else:
            initial_sleep_period = 0
        rates = np.full(self.n_models, initial_tcs)  # Initial inter-flash intervals for all models
        rate_scale = self.default_rate_scale
        poisson_base_rates = rates * rate_scale  # Base lambda rates for Poisson
        individual_dropout_probs = np.random.poisson(poisson_base_rates) / 100
        individual_dropout_probs = np.clip(individual_dropout_probs, 0, 1)  # Ensure valid probabilities
        base_dropout_probs = individual_dropout_probs
        states = np.full(self.n_models, 'integrate', dtype='<U10')  # States for all models
        flash_counts = np.full(self.n_models, initial_fcs)
        flash_counts = np.ceil(np.sqrt(flash_counts)).astype(int)

        #  Model dynamics loop
        V = np.zeros((self.n_models, len(t)))
        V[:, 0] = np.random.uniform(0.33, 0.66, self.n_models)
        for i in range(1, len(t)):
            for j in range(self.n_models):
                if states[j] == 'integrate':
                    rate = (self.dt / rates[j])
                    if self.log:
                        rate = (((math.log(2) / rates[j]) * (1 - V[j, i - 1])) * self.dt) + ((self.dt / rates[j]) ** 2)
                    V[j, i] = V[j, i - 1] + rate
                    if not self.driven:
                        if V[j, i] >= 1:
                            if individual_dropout_probs[j] < np.random.random():
                                flash_counts[j] -= 1
                                V[j, i] = 1.0
                                states[j] = 'reset'
                                rate_scale += 2
                                individual_dropout_probs[j] = np.clip(np.random.poisson(
                                    rates[j] * rate_scale) / 100, 0, 1)
                            else:
                                V[j, i] = 0.0
                                individual_dropout_probs[j] = base_dropout_probs[j]
                                rate_scale = self.default_rate_scale
                    else:
                        if V[j, i] >= 1:
                            if j == self.driver:
                                if i < initial_sleep_period:
                                    V[j, i] = 0.0
                                    continue
                                else:
                                    V[j, i] = 1.0
                                    states[j] = 'reset'

                            else:
                                if individual_dropout_probs[j] < np.random.random():
                                    flash_counts[j] -= 1
                                    V[j, i] = 1.0
                                    states[j] = 'reset'
                                    rate_scale += 2
                                    individual_dropout_probs[j] = np.clip(np.random.poisson(
                                        rates[j] * rate_scale) / 100, 0, 1
                                                                          )
                                else:
                                    V[j, i] = 0.0
                                    individual_dropout_probs[j] = base_dropout_probs[j]
                                    rate_scale = self.default_rate_scale

                elif states[j] == 'reset':
                    V[j, i] = V[j, i - 1] - (self.dt / self.td)
                    if V[j, i] <= 0:
                        V[j, i] = 0.0
                        states[j] = 'integrate'
                        rates[j] = initial_tcs[0][j]
                        if flash_counts[j] <= 0:
                            if self.driven:
                                if j != self.driver:
                                    rates[j] = math.sqrt(self.ib_df.sample().values[0])
                                    flash_counts[j] = np.ceil(np.sqrt(self.fc_df.sample().values[0]).astype(int))
                            else:
                                self.attns[j] = self.attn
                                rates[j] = math.sqrt(self.ib_df.sample().values[0])
                                flash_counts[j] = np.ceil(np.sqrt(self.fc_df.sample().values[0]).astype(int))

                # Coupling from other flashers
                # If sight lines are established do coupling, else skip
                if 0.0 <= self.attn:  # self.w_p[i] < self.attn:
                    active_flashers = len([x for x in states if x != 'integrate'])
                    # go through all other individuals
                    for indiv in range(self.n_models):
                        if self.driven:
                            # If j (acting individual) is the driver, it does not receive coupling
                            if j == self.driver:
                                V[j, i] = min(V[j, i], 1.0)
                                V[j, i] = max(V[j, i], 0.0)
                            else:
                                # If j (acting individual) is not the neighbor indiv, it receives coupling if
                                # a) indiv is resetting (flashing)
                                # b) j (acting individual) is integrating (charging)
                                if indiv != j and states[indiv] == 'reset' and states[j] == 'integrate':
                                    if ks[j] <= V[j, i] <= 1:
                                        V[j, i] += (b / active_flashers)
                                        V[j, i] = min(V[j, i], 1.0)
                                    elif 0 <= V[j, i] < ks[j]:
                                        if ms == 'E':
                                            V[j, i] += (b / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)
                                        elif ms == 'ER':
                                            V[j, i] += (0 / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)
                                        else:  # EI
                                            V[j, i] -= (b / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)
                        else:
                            if self.connectivity_graph:
                                if indiv != j and states[indiv] == 'reset' and states[j] != 'reset':
                                    indiv_coord = self.map_to_2d(indiv)
                                    j_coord = self.map_to_2d(j)

                                    self.connectivity_graph.add_edge(indiv_coord, j_coord)
                                    if ks[j] <= V[j, i] <= 1:
                                        V[j, i] += (b / active_flashers)
                                        V[j, i] = min(V[j, i], 1.0)
                                    elif 0 <= V[j, i] < ks[j]:
                                        if ms == 'E':
                                            V[j, i] += (b / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)
                                        elif ms == 'ER':
                                            V[j, i] += (0 / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)
                                        else:  # EI
                                            V[j, i] -= (b / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)
                            else:
                                if indiv != j and states[indiv] == 'reset' and states[j] != 'reset':
                                    if ks[j] <= V[j, i] <= 1:
                                        V[j, i] += (b / active_flashers)
                                        V[j, i] = min(V[j, i], 1.0)
                                    elif 0 <= V[j, i] < ks[j]:
                                        if ms == 'E':
                                            V[j, i] += (b / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)
                                        elif ms == 'ER':
                                            V[j, i] += (0 / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)
                                        else:  # EI
                                            V[j, i] -= (b / active_flashers)
                                            V[j, i] = max(V[j, i], 0.0)

        # stat keeping
        vs = []
        for j in range(self.n_models):
            v = [0 if x < 1 else 1 for x in V[j]]
            vs.append(v)

        statkeeping = {'beta': b, 'k_thresh': self.k, 'n_models': self.n_models, 'driven_freq': self.driven_freq}
        for j, v_trace in enumerate(vs):
            ints = analysis.calculate_intervals(v_trace)
            intfs = [x / 100 for x in ints if 0.25 < x / 100 < 2.0]
            intbs = [x / 100 for x in ints if x / 100 >= 2.0]
            statkeeping.update({'interflashes_{}'.format(j): intfs})
            statkeeping.update({'interbursts_{}'.format(j): intbs})
            statkeeping.update({'spiketimes_{}'.format(j): np.where(np.array(v_trace) == 1.0)[0]})

        self.stats = statkeeping
        self.v_results = V
        self.time_vector = t
        self.status = True


def parse_betas(input_string):
    if input_string:
        return list(map(float, input_string.split(',')))
    else:
        return []


def parse_freqs(input_string):
    if input_string:
        return list(map(float, input_string.split(',')))
    else:
        return []


def parse_ks(input_string):
    if input_string:
        return list(map(float, input_string.split(',')))
    else:
        return []


def load_data(args):
    # Define different initial interflash intervals for each model
    with open(args.ib_data_fpath, 'r') as f:
        ib_df = pd.read_csv(f, header=None)
    ib_df.columns = ['interval']
    ib_df = ib_df[ib_df.interval <= 10]

    # Define different initial interflash intervals for each model
    with open(args.fc_data_fpath, 'r') as f:
        fc_df = pd.read_csv(f)

    with open(args.if_data_fpath, 'r') as f:
        if_df = pd.read_csv(f, header=None)
    if_df.columns = ['interval']
    if_df = if_df[if_df.interval > 0.3]
    return ib_df, if_df, fc_df


def setup_params(args, dt):
    if args.betas is None or len(args.betas) == 0:
        args.betas = [0.5]
    if args.ks is None or len(args.ks) == 0:
        args.ks = [0.5]
    if args.driven_freq is None or len(args.driven_freq) == 0:
        args.driven_freq = [None]
    betas = args.betas
    ks = args.ks
    driven_freqs = args.driven_freq
    n_models = args.n  # Number of interacting models
    n_trials = args.n_trials
    log = args.log
    if args.driven:
        driver = 0
        initial_sleep_period = 60 / dt
    else:
        driver = -1
        initial_sleep_period = 0
    tT = args.total_t  # Total time (s)
    td = args.fl

    return betas, ks, tT, td, n_models, n_trials, log, driver, initial_sleep_period, driven_freqs


def run_sim(sim):
    sim.simulate()
    return sim


def load_args():
    parser = argparse.ArgumentParser(
        prog='Two-timescale IF Model',
        description='Implements two-timescale integrate-and-fire model on any number of connected agents',
    )
    parser.add_argument('--model_specifics', type=str, default='EI',
                        help='Choose from one of 3 models: excitatory (E), '
                             'excitatory-inhibitory (EI), '
                             'excitatory-refractory(ER)'
                        )
    parser.add_argument('--fc_data_fpath', type=str, default='data/frontalis_flashcounts.csv',
                        help='Path to the flash count distribution of P. frontalis from data')
    parser.add_argument('--if_data_fpath', type=str, default='data/frontalis_interflash.csv',
                        help='Path to the interflash interval distribution of P. frontalis from data')
    parser.add_argument('--ib_data_fpath', type=str, default='data/frontalis_interburst.csv',
                        help='Path to the interburst interval distribution of P. frontalis from data')
    parser.add_argument('--driven', action='store_true',
                        help='Whether to drive the dynamics with a LED mimic')
    parser.add_argument('--driven_freq',
                        type=parse_freqs,
                        default=None,
                        help='Comma-separated list of floats = driven frequency values in seconds. Defaults to 0.6')
    parser.add_argument('--total_t', type=int, default=150,
                        help='Total simulation time (seconds)')
    parser.add_argument('--attention_rate', type=float, default=0.833,
                        help='Percentage of time spent looking at the driving signal')
    parser.add_argument('--fl', type=float, default=0.033,
                        help='Flash length from data (seconds)')
    parser.add_argument('--n', type=int, default=2,
                        help='Number of individuals to simulate')
    parser.add_argument('--n_trials', type=int, default=1,
                        help='Number of trials to per parameter set')
    parser.add_argument('--ks',
                        type=parse_ks,
                        default=None,
                        help='Comma-separated list of floats = refractory threshold parameter values. Defaults to 0.5')
    parser.add_argument('--log', action='store_true',
                        help='Whether to use logarithmic charging')
    parser.add_argument('--betas',
                        type=parse_betas,
                        default=None,
                        help='Comma-separated list of floats = beta parameter values. Defaults to 0.5'
                        )
    parser.add_argument('--visualize_simulation', action='store_true',
                        help='Whether to visualize time series from simulation instance(s)')
    parser.add_argument('--on_grid', action='store_true',
                        help='Whether to run on a predefined grid embedding')
    parser.add_argument('--scale_same', action='store_true',
                        help='Whether to set initial interflash distribution to be the same')
    parser.add_argument('--save_folder',
                        default=os.getcwd(),
                        help='Where to save results, default is the current working dir')

    a = parser.parse_args()
    return a


def main():
    args = load_args()

    # Parameters
    dt = 0.01  # Time step (s)
    betas, ks, tT, td, n_models, n_trials, log, driver, initial_sleep_period, driven_freqs = setup_params(args, dt)
    ib_df, if_df, fc_df = load_data(args)

    processes = []
    for beta in betas:
        for k in ks:
            for driven_freq in driven_freqs:
                for trial in range(n_trials):
                    p = Simulation(trial, beta, k, if_df, fc_df, ib_df, td, args.driven, driver, driven_freq, n_models,
                                   tT, dt,
                                   args.model_specifics,
                                   args.log,
                                   args.attention_rate,
                                   args.on_grid,
                                   args.scale_same)
                    processes.append(p)

    process_pool = Pool(int(multiprocessing.cpu_count() / 2))
    process_results = process_pool.map(run_sim, processes)

    final_results = [pr for pr in process_results]
    final_stats = [pr.stats for pr in process_results]
    if args.visualize_simulation:
        for i,fr in enumerate(final_results):
            analysis.plots(fr.n_models, fr.v_results, fr.time_vector, fr.beta, fr.k, i)

    dists_df = pd.DataFrame(final_stats)
    dists_df.to_csv('{}/dists.csv'.format(args.save_folder), index=False)


if __name__ == "__main__":
    main()
