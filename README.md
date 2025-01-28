Multiprocessed simulation of parameter sweeps for various integrate and fire models of P.frontalis synchronization

Implements two-timescale integrate-and-fire model on any number of connected agents
Additionally implements comparative analysis between simulation results and results from driven experiments on P. frontalis fireflies. 
Data from experiments is housed in data/all_afters_from_experiments.pickle, as well as in raw form at [this accompanying GitHub repository](www.github.com/owingit/led_firesync)


Simulation: 
Two-timescale IF Model [-h] [--model_specifics MODEL_SPECIFICS] [--fc_data_fpath FC_DATA_FPATH] [--if_data_fpath IF_DATA_FPATH]
                              [--ib_data_fpath IB_DATA_FPATH] [--driven] [--driven_freq DRIVEN_FREQ] [--total_t TOTAL_T] [--fl FL] [--n N] [--n_trials N_TRIALS]
                              [--ks KS] [--log] [--betas BETAS] [--save_folder SAVE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  
  --model_specifics MODEL_SPECIFICS
                        Choose from one of 3 models: excitatory (E), excitatory-inhibitory (EI), excitatory-refractory(ER)
  
  --fc_data_fpath FC_DATA_FPATH
                        Path to the flash count distribution of P. frontalis from data
  
  --if_data_fpath IF_DATA_FPATH
                        Path to the interflash interval distribution of P. frontalis from data
  
  --ib_data_fpath IB_DATA_FPATH
                        Path to the interburst interval distribution of P. frontalis from data
  
  --driven              Whether to drive the dynamics with a LED mimic
  
  --driven_freq DRIVEN_FREQ
                        Comma-separated list of floats = driven frequency values in seconds. Defaults to 0.6
  
  --total_t TOTAL_T     Total simulation time (seconds)
  
  --fl FL               Flash length from data (seconds)
  
  --n N                 Number of individuals to simulate
  
  --n_trials N_TRIALS   Number of trials to per parameter set
  
  --ks KS               Comma-separated list of floats = refractory threshold parameter values. Defaults to 0.5
  
  --log                 Whether to use logarithmic charging
  
  --betas BETAS         Comma-separated list of floats = beta parameter values. Defaults to 0.5
  
  --save_folder SAVE_FOLDER
                        Where to save results, default is the current working dir


Default data paths point to the files in the data/ repository. 

Analysis usage:
python analyze_results.py --plot 
