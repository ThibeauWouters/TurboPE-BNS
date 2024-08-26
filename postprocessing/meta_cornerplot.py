import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import corner
import h5py
import jax.numpy as jnp
import jax
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
jax.config.update("jax_disable_jit", True)
import json
import copy
from scipy.spatial.distance import jensenshannon
import pickle

from ripple import get_chi_eff, Mc_eta_to_ms, lambda_tildes_to_lambdas

import utils_compare_runs
from utils_compare_runs import paths_dict, jim_naming, LABELS

import seaborn as sns
import pandas as pd

COLORS = {"pbilby": "blue", 
          "jim": "red", 
          "RB": "green", 
          "ROQ": "brown"}

def main():
    
    # Plot all on top of each other
    
    plt.figure() 
    
    all_events = ["GW170817_TaylorF2",
                  "GW170817_NRTidalv2",
                  "GW190425_TaylorF2",
                  "GW190425_NRTidalv2"]
    
    for i, event in enumerate(all_events):
        # Pbilby
        path = paths_dict[event]["bilby"]
        chains = utils_compare_runs.get_chains_bilby(path)
        
        if  i == 0:
            fig = corner.corner(chains, labels=LABELS, color=COLORS["pbilby"])
        else:
            corner.corner(chains, fig=fig, color=COLORS["pbilby"])
            
        # Jim:
        path = paths_dict[event]["jim"]
        chains = utils_compare_runs.get_chains_jim(path)
        corner.corner(chains, fig=fig, color=COLORS["jim"])
        
        # RB:
        path = utils_compare_runs.bilby_RB_paths_dict[event]
        chains = utils_compare_runs.get_chains_hdf5(path)
        corner.corner(chains, fig=fig, color=COLORS["RB"])
        
        # ROQ: 
        try: 
            path = utils_compare_runs.bilby_ROQ_paths_dict[event]
            chains = utils_compare_runs.get_chains_hdf5(path)
            corner.corner(chains, fig=fig, color=COLORS["ROQ"])
        except Exception as e:
            print("This event is not supported by ROQ yet. Full error message follows now:")
            print(e)
        
        plt.savefig(f"../figures/meta_corner_{event}.png")
        plt.close()
    
if __name__ == "__main__":
    main()