import numpy as np
import math
import os
import pandas as pd

import multiprocessing
from multiprocessing import Manager, Process, Pool

import argparse

import analysis


class Simulation:
    def __init__(self, trial_num, trial_beta, trial_k, if_df, fc_df, ib_df, trial_td, trial_driven, trial_driver,
                 trial_driven_freq, trial_n_models, trial_T, dt, model_specifics, trial_log):
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
        self.stats = None

    def simulate(self):
        b = self.beta

        ms = self.model
        print('Trial {} - k {} - beta {} - driven freq {}'.format(self.trial_num,
                                                                  self.k,
                                                                  self.beta,
                                                                  self.driven_freq))
        t = np.arange(0, self.T, self.dt)  # Time vector
        initial_tcs = self.if_df.sample(self.n_models, ignore_index=True).to_numpy().reshape(1, self.n_models)
        initial_fcs = self.fc_df.sample(self.n_models, ignore_index=True).to_numpy().reshape(1, self.n_models)
        if self.driven:
            initial_tcs[0][self.driver] = self.driven_freq
            initial_fcs[0][self.driver] = len(t) - 1  # never runs out
            initial_sleep_period = 60 / self.dt
        else:
            initial_sleep_period = 0
        rates = np.full(self.n_models, initial_tcs)  # Initial inter-flash intervals for all models
        states = np.full(self.n_models, 'integrate', dtype='<U10')  # States for all models
        flash_counts = np.full(self.n_models, initial_fcs)

        #  Model dynamics
        V = np.zeros((self.n_models, len(t)))
        V[:, 0] = np.random.rand(self.n_models)
        for i in range(1, len(t)):
            assert (rates[0] == self.driven_freq)
            for j in range(self.n_models):
                if states[j] == 'integrate':
                    rate = (self.dt / rates[j])
                    if self.log:
                        rate = (((math.log(2) / rates[j]) * (1 - V[j, i - 1])) * self.dt) + ((self.dt / rates[j]) ** 2)
                    V[j, i] = V[j, i - 1] + rate
                    if not self.driven:
                        if V[j, i] >= 1:
                            flash_counts[j] -= 1
                            V[j, i] = 1.0
                            states[j] = 'reset'
                    else:
                        if V[j, i] >= 1:
                            if j == self.driver and i < initial_sleep_period:
                                V[j, i] = 0.0
                                continue
                            else:
                                flash_counts[j] -= 1
                                V[j, i] = 1.0
                                states[j] = 'reset'

                elif states[j] == 'reset':
                    V[j, i] = V[j, i - 1] - (self.dt / self.td)
                    if V[j, i] <= 0:
                        V[j, i] = 0.0
                        states[j] = 'integrate'
                        rates[j] = initial_tcs[0][j]
                        if flash_counts[j] <= 0:
                            if self.driven:
                                if j != self.driver:
                                    rates[j] = self.ib_df.sample().values[0]
                                    flash_counts[j] = self.fc_df.sample().values[0]
                            else:
                                rates[j] = self.ib_df.sample().values[0]
                                flash_counts[j] = self.fc_df.sample().values[0]

                # Coupling from other flashers
                active_flashers = len([x for x in states if x != 'integrate'])
                for indiv in range(self.n_models):
                    if self.driven:
                        if j == self.driver:
                            V[j, i] = min(V[j, i], 1.0)
                            V[j, i] = max(V[j, i], 0.0)
                        else:
                            if indiv != j and states[indiv] == 'reset' and states[j] == 'integrate':
                                if self.k <= V[j, i] <= 1:
                                    V[j, i] += (b / active_flashers)
                                    V[j, i] = min(V[j, i], 1.0)
                                elif 0 <= V[j, i] < self.k:
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
                            if self.k <= V[j, i] <= 1:
                                V[j, i] += (b / active_flashers)
                                V[j, i] = min(V[j, i], 1.0)
                            elif 0 <= V[j, i] < self.k:
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
        # if n_models == 2:
        #     percentage = analysis.calculate_percentage(vs[0], vs[1], window_size=int(5 / dt))

        statkeeping = {'beta': b, 'k_thresh': self.k, 'n_models': self.n_models, 'driven_freq': self.driven_freq}
        for j, v_trace in enumerate(vs):
            ints = analysis.calculate_intervals(v_trace)
            intfs = [x / 100 for x in ints if 0.25 < x / 100 < 2.0]
            intbs = [x / 100 for x in ints if x / 100 >= 2.0]
            statkeeping.update({'interflashes_{}'.format(j): intfs})
            statkeeping.update({'interbursts_{}'.format(j): intbs})
            statkeeping.update({'spiketimes_{}'.format(j): np.where(np.array(v_trace) == 1.0)[0]})
        self.stats = statkeeping
        # dists_df = dists_df.append(
        #         statkeeping, ignore_index=True
        #     )
        # analysis.plots(n_models, V, t, beta)


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
        args.driven_freq = [0.6]
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


def main():
    parser = argparse.ArgumentParser(
        prog='Two-timescale IF Model',
        description='Implements two-timescale integrate-and-fire model on any number of connected agents',
    )
    parser.add_argument('--model_specifics', type=str, default='EI',
                        help='Choose from one of 3 models: excitatory (E), '
                             'excitatory-inhibitory (EI), '
                             'excitatory-refractory(ER)'
                        )
    parser.add_argument('--fc_data_fpath', type=str, default='data/frontalis_flashcounts.csv')
    parser.add_argument('--if_data_fpath', type=str, default='data/frontalis_interflash.csv')
    parser.add_argument('--ib_data_fpath', type=str, default='data/frontalis_interburst.csv')
    parser.add_argument('--driven', action='store_true')
    parser.add_argument('--driven_freq', type=parse_freqs, default=None,
                        help='Comma-separated list of beta value floats')
    parser.add_argument('--total_t', type=int, default=600)
    parser.add_argument('--fl', type=float, default=0.03)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--ks', type=parse_ks,
                        help='Comma-separated list of beta value floats',
                        default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--betas',
                        type=parse_betas,
                        help='Comma-separated list of beta value floats',
                        default=None
                        )
    parser.add_argument('--save_folder',
                        default=os.getcwd())

    args = parser.parse_args()

    # Parameters
    dt = 0.01  # Time step (s)
    betas, ks, tT, td, n_models, n_trials, log, driver, initial_sleep_period, driven_freqs = setup_params(args, dt)
    ib_df, if_df, fc_df = load_data(args)

    distlist = ['beta', 'k_thresh', 'n_models', 'driven_freq']
    for model_n in range(n_models):
        distlist.append('interflashes_{}'.format(model_n))
        distlist.append('interbursts_{}'.format(model_n))
        distlist.append('spiketimes_{}'.format(model_n))

    processes = []
    for trial in range(n_trials):
        for beta in betas:
            for k in ks:
                for driven_freq in driven_freqs:
                    p = Simulation(trial, beta, k, if_df, fc_df, ib_df, td, args.driven, driver,
                                   driven_freq, n_models, tT, dt, args.model_specifics, args.log)
                    processes.append(p)

    process_pool = Pool(int(multiprocessing.cpu_count() / 2))
    process_results = process_pool.map(run_sim, processes)

    final_results = [pr.stats for pr in process_results]

    dists_df = pd.DataFrame(final_results)
    dists_df.to_csv('{}/dists.csv'.format(args.save_folder), index=False)


if __name__ == "__main__":
    main()
