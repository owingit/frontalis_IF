import pickle

import argparse

import analysis


def load_from_csvs(cp):
    p = cp[0]
    e_r = analysis.load_parameter_sweep_data(p)
    print('loaded E')

    p = cp[1]
    er_r = analysis.load_parameter_sweep_data(p)
    print('loaded ER')

    p = cp[2]
    ei_r = analysis.load_parameter_sweep_data(p)
    print('loaded EI')
    return e_r, er_r, ei_r


def load_from_pickles(pp):
    p = pp[0]
    with open(p, 'rb') as f:
        e_r = pickle.load(f)
    print('loaded E')
    p = pp[1]
    with open(p, 'rb') as f:
        er_r = pickle.load(f)
    print('loaded ER')
    p = pp[2]
    with open(p, 'rb') as f:
        ei_r = pickle.load(f)
    print('loaded EI')
    return e_r, er_r, ei_r


def main():
    parser = argparse.ArgumentParser(
        prog='Two-timescale IF Model ANALYSIS module',
        description='Implements analysis of results of integrate-and-fire model compared with experiments',
    )
    parser.add_argument('--use_csvs', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--output', type=str, default='all_afters_from_experiments')
    args = parser.parse_args()

    with open('{}/{}.pickle'.format(args.data_folder, args.output), 'rb') as f:
        output_dist = pickle.load(f)
    print('loaded output')
    led_freqs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.77, 0.85, 1.0]
    expected_afters = {}
    for led_freq in led_freqs:
        expected_afters[led_freq] = output_dist[str(int(led_freq * 1000))]

    csv_paths = ['{}/dists_E.csv'.format(args.data_folder),
                 '{}/dists_ER.csv'.format(args.data_folder),
                 '{}/dists_EI.csv'.format(args.data_folder)]
    pickle_paths = ['{}/Excitatory_df_combined_0.pickle'.format(args.data_folder),
                    '{}/Excitatory-refractory_df_combined_0.pickle'.format(args.data_folder),
                    '{}/Excitatory-inhibitory_df_combined_0.pickle'.format(args.data_folder)]
    if args.use_csvs:
        excitatory_results, excitatory_refractory_results, excitatory_inhibitory_results = load_from_csvs(csv_paths)
    else:
        excitatory_results, excitatory_refractory_results, excitatory_inhibitory_results = load_from_pickles(pickle_paths)

    models = [('Excitatory', excitatory_results),
              ('Excitatory-refractory', excitatory_refractory_results),
              ('Excitatory-inhibitory', excitatory_inhibitory_results)]

    analysis.compare_models(expected_afters, models, plot=args.plot)
    print('Done')


if __name__ == "__main__":
    main()
