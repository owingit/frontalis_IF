import numpy as np
import math
import random
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Manager, Process, Pool
import seaborn as sns

import argparse

import analysis

p = 'driven_res/dists_E_100.csv'
excitatory_results = analysis.load_parameter_sweep_data(p)

p = 'driven_res/dists_ER_100.csv'
excitatory_refractory_results = analysis.load_parameter_sweep_data(p)

p = 'driven_res/dists_EI_100.csv'
excitatory_inhibitory_results = analysis.load_parameter_sweep_data(p)

with open('driven_res/all_afters_from_experiments_20240607.pickle', 'rb') as f:
    output_dist = pickle.load(f)
expected_afters = {}
for led_freq in [0.3, 0.4, 0.5, 0.6, 0.7, 0.77, 0.85, 1.0]:
    expected_afters[led_freq] = output_dist[str(int(led_freq * 1000))]

models = [('Excitatory', excitatory_results),
          ('Excitatory-refractory', excitatory_refractory_results),
          ('Excitatory-inhibitory', excitatory_inhibitory_results)]

analysis.compare_models(expected_afters, models, plot=True)
print('here')
