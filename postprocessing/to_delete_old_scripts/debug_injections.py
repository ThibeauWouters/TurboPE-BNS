import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import pickle
import numpy as np
import json
from scipy.stats import percentileofscore, uniform, kstest
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from ripple import get_chi_eff, Mc_eta_to_ms
from tqdm import tqdm

fs = 26
matplotlib_params = {"axes.grid": True,
          "text.usetex" : True,
          "font.family" : "serif",
          "ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "font.serif" : ["Computer Modern Serif"],
          "xtick.labelsize": fs,
          "ytick.labelsize": fs,
          "axes.labelsize": fs,
          "legend.fontsize": fs,
          "legend.title_fontsize": fs,
          "figure.titlesize": fs}
plt.rcParams.update(matplotlib_params)

################
### PREAMBLE ###
################


outdirs_dict = {"TF2": '/home/thibeau.wouters/TurboPE-BNS/injections/outdir_TF2/',
                "NRTv2": '/home/thibeau.wouters/TurboPE-BNS/injections/outdir_NRTv2/'}

problematic_injections_dict = {"NRTv2": [1, 2, 45, 74, 81, 89],
                               "TF2": []}
naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

def main():
    
    name = "TF2"
    print("Name waveform: ", name)
    outdir = outdirs_dict[name]
    
    which_index = naming.index("ra")
    
    results_dict = {}
    
    for subdir in tqdm(os.listdir(outdir)):
        # Check if directory exists:
        if not os.path.isdir(f"{outdir}/{subdir}"):
            continue
        
        # Check if it has results_production.npz:
        if not os.path.isfile(f"{outdir}/{subdir}/results_production.npz"):
            continue
        
        # Loading data
        with open(f'{outdir}/{subdir}/config.json') as f:
            config = json.load(f)
            
        true_params = [config[name] for name in naming]
        
        all_posterior_samples = np.load(f'{outdir}/{subdir}/results_production.npz')
        all_posterior_samples = all_posterior_samples['chains'].T
        all_posterior_samples = [all_posterior_samples[param_idx].flatten() for param_idx in range(len(naming))]
        
        # Focus on one parameter
        posterior_samples = all_posterior_samples[which_index]
        true_param = true_params[which_index]
        
        p_value = percentileofscore(posterior_samples, true_param) / 100.
        
        results_dict[subdir] = p_value

    # Sort the results from low p value to high p value
    results_dict = sorted(results_dict.items(), key=lambda item: item[1])

    for key, value in results_dict:
        print(f"{key}: {value}")

    return subdir

if __name__ == "__main__":
    main()
    print("Done!!!")