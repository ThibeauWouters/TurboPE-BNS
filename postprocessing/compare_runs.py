"""
Script to load and compare jim runs against Bilby runs
Here we compare our posterior samples with those obtained from GWOSC.

More information:
- GWOSC page for GW190425 results: https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190425/v3/
- Public posterior samples can be found here: https://dcc.ligo.org/public/0165/P2000026/002/posterior_samples.h5
"""

import psutil
p = psutil.Process()
p.cpu_affinity([0])

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

################
### PREAMBLE ###
################

params = {
    # "axes.labelsize": 132,
    # "axes.titlesize": 132,
    "text.usetex": True,
    "font.family": "times new roman",
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
plt.rcParams.update(params)

label_fontsize = 26
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=label_fontsize),
                        title_kwargs=dict(fontsize=label_fontsize), 
                        color="blue",
                        # quantiles=[],
                        levels=[0.68, 0.95], # 0.997 # for 3 sigma as well?
                        plot_density=False,
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        labelpad = 0.25
)

my_gray = "#ababab"
my_blue = "#0f7db5"
my_colors = {"jim": my_blue, 
             "bilby": my_gray,
             "jim_no_taper": "#9b3460",} # "#dd4a89"
histogram_fill_color = "#d6d6d6"

labels_chi_eff = [r'$\mathcal{M}/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$' ,r'$d_{\rm{L}}/{\rm Mpc}$',r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']


#################
### UTILITIES ###
#################

def weight_function(x):
    return x**2

def reweigh_distance(chains, d_idx = 6):
    """
    Get weights based on distance to mimic cosmological distance prior.
    """
    d_samples = chains[:, d_idx]
    weights = weight_function(d_samples)
    weights = weights / np.sum(weights)
    
    return weights

####################
### LOADING DATA ###
####################


def get_plot_samples(posterior_list: list[np.array],
                     idx_list: list[int] = []):
    
    plotsamples_list = []
        
    sizes = [len(posterior) for posterior in posterior_list]
    smallest_size = min(sizes)
    
    # Resample posteriors for the same size
    for i, posterior in enumerate(posterior_list):
        this_size = sizes[i]
        all_idx = np.arange(this_size)
        sampled_idx = np.random.choice(all_idx, size=smallest_size, replace=False)
        sampled_posterior = posterior[sampled_idx]
        plotsamples_list.append(sampled_posterior)
        
    # plotsamples_dummy = np.vstack([plotsamples_list[idx_list[i]] for i in range(len(idx_list))])
    plotsamples_dummy = np.empty_like(plotsamples_list[0])
    print(np.shape(plotsamples_dummy))
    plotsamples_list = np.array(plotsamples_list)
    for i in range(len(idx_list)):
        plotsamples_dummy[:, i] = plotsamples_list[idx_list[i], :, i]
    
    return plotsamples_list, plotsamples_dummy



def plot_comparison(jim_samples: np.array, 
                    bilby_samples: np.array, 
                    idx_list: list,
                    use_weights = False,
                    save: bool = True,
                    save_name = "corner_comparison",
                    which_waveform: str = "TaylorF2",
                    remove_tc: bool = True,
                    convert_chi: bool = True,
                    convert_lambdas: bool = False,
                    corner_kwargs: dict = {},
                    **kwargs):
    
    # If wanted, reweigh uniform samples to powerlaw samples
    if use_weights:
        weights = reweigh_distance(jim_samples)
        save_name += "_reweighted"
    
    if not use_weights:
        weights = None
    
    jim_samples = utils_compare_runs.preprocess_samples(jim_samples, convert_chi = convert_chi, convert_lambdas = convert_lambdas)
    bilby_samples = utils_compare_runs.preprocess_samples(bilby_samples, convert_chi = convert_chi, convert_lambdas = convert_lambdas)
    
    # Drop NaNs:
    nan_idx_list = np.argwhere(np.isnan(bilby_samples))
    nan_idx_list = np.unique(nan_idx_list[:, 0])
    bilby_samples = np.delete(bilby_samples, nan_idx_list, axis=0)
    
    # Drop infs:
    inf_idx_list = np.argwhere(np.isinf(bilby_samples))
    inf_idx_list = np.unique(inf_idx_list[:, 0])
    bilby_samples = np.delete(bilby_samples, inf_idx_list, axis=0)
    
    # Same for jim samples:
    nan_idx_list = np.argwhere(np.isnan(jim_samples))
    nan_idx_list = np.unique(nan_idx_list[:, 0])
    jim_samples = np.delete(jim_samples, nan_idx_list, axis=0)
    
    # Drop infs:
    inf_idx_list = np.argwhere(np.isinf(jim_samples))
    inf_idx_list = np.unique(inf_idx_list[:, 0])
    jim_samples = np.delete(jim_samples, inf_idx_list, axis=0)
    
    # Remove the t_c label for comparison with bilby
    if remove_tc:
        labels = copy.deepcopy(LABELS)
        labels.remove(r'$t_c$')
    else:
        labels = LABELS
        
    if convert_lambdas:
        labels = copy.deepcopy(labels)
        labels.remove(r'$\Lambda_1$')
        labels.remove(r'$\Lambda_2$')
        labels.insert(4, r'$\tilde{\Lambda}$')
        labels.insert(5, r'$\delta\tilde{\Lambda}$')
    
    if convert_chi:
        labels = copy.deepcopy(labels)
        labels.remove(r'$\chi_1$')
        labels.remove(r'$\chi_2$')
        labels.insert(2, r'$\chi_{\rm eff}$')
        
    if "170817" in save_name:
        labelpad = 0.25
    else:
        labelpad = 0.1
    corner_kwargs["labelpad"] = labelpad
    
    print(f"Saving plot of chains to {save_name}")
    plotsamples_list, dummy_values = get_plot_samples([jim_samples, bilby_samples], idx_list)
    
    # Go over to selecting kwargs:
    first_color = kwargs["first_color"] if "first_color" in kwargs else my_colors["jim"]
    second_color = kwargs["second_color"] if "second_color" in kwargs else my_colors["bilby"]
    
    first_label = kwargs["first_label"] if "first_label" in kwargs else r'\textsc{Jim}'
    second_label = kwargs["second_label"] if "second_label" in kwargs else r'\textsc{pBilby}'
    histogram_fill_color = kwargs["histogram_fill_color"] if "histogram_fill_color" in kwargs else "#d6d6d6"
    
    # Actual plotting
    hist_kwargs={'density': True, 'linewidth': 1.5}
    contour_lw = 2
    for i, (samples, color) in enumerate(zip(plotsamples_list, [first_color, second_color])):
        corner_kwargs["color"] = color
        if i == 1:
            # bilby kwargs
            zorder = 10
            
            corner_kwargs["fill_contours"] = True
            corner_kwargs["plot_contours"] = True
            corner_kwargs["zorder"] = zorder
            corner_kwargs["alpha"] = 1
            corner_kwargs["contour_kwargs"] = {"zorder": zorder,
                                               "linewidths": contour_lw}
            corner_kwargs["contourf_kwargs"] = {"zorder": zorder}
            
            hist_kwargs["fill"] = True
            hist_kwargs["facecolor"] = histogram_fill_color
            hist_kwargs["zorder"] = zorder
            hist_kwargs["color"] = color
        else:
            # jim kwargs
            zorder = 1e9
            corner_kwargs["plot_contours"] = True
            corner_kwargs["fill_contours"] = False
            corner_kwargs["no_fill_contours"] = True
            corner_kwargs["alpha"] = 1
            corner_kwargs["contour_kwargs"] = {"zorder": zorder, 
                                               "linewidths": contour_lw}
            corner_kwargs["contourf_kwargs"] = {"zorder": zorder}
            
            hist_kwargs["fill"] = False
            hist_kwargs["zorder"] = zorder
            hist_kwargs["color"] = color
        
        corner_kwargs["hist_kwargs"] = hist_kwargs
        corner_kwargs["density"] = True
        if i == 0:
            fig = corner.corner(samples, labels = labels, weights=weights, **corner_kwargs)
        else:
            corner.corner(samples, labels = labels, fig=fig, weights=weights, **corner_kwargs)
    
    # Dummy postprocessing
    corner_kwargs["color"] = "white"
    corner_kwargs["plot_contours"] = True
    corner_kwargs["no_fill_contours"] = True
    corner_kwargs["alpha"] = 0
    corner_kwargs["contour_kwargs"] = {"zorder": 0.00001, 
                                        "linewidths": 0.0,
                                        "alpha": 0.0}
    corner_kwargs["contourf_kwargs"] = {"zorder": 0.00001}
    
    hist_kwargs={'density': True, 'alpha': 0}
    corner_kwargs["hist_kwargs"] = hist_kwargs
    corner.corner(dummy_values, fig=fig, **corner_kwargs)
    
    # Add a custom legend
    legend_lw = 2
    legend_elements = [
        mlines.Line2D([], [], color=first_color, linewidth = legend_lw, label=first_label),
        mlines.Line2D([], [], color=second_color, linewidth = legend_lw, label=second_label)
    ]

    # Create the legend
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.925, 0.925), fontsize=40, frameon = False) 
    
    if save:
        for ext in ["png", "pdf"]:
            plt.savefig(f"{save_name}.{ext}", bbox_inches='tight')
    plt.show()
    plt.close()
    

def compare_bilby_runs(peter_path: str, 
                       gwosc_path: str, 
                       save_name = "bilby_comparison",
                       which_waveform: str = "TaylorF2",
                       remove_tc : bool = True):
    """
    Compare Peter's samples with the publicly released samples.

    Args:
        bilby_path_1 (_type_): _description_
        bilby_path_2 (_type_): _description_
        save_name (str, optional): _description_. Defaults to "corner_comparison".
    """
    
    # Load Peter's samples
    peter_samples = utils_compare_runs.get_chains_bilby(peter_path)
    
    # Load GWOSC samples
    gwosc_samples = utils_compare_runs.get_chains_GWOSC(gwosc_path, which_waveform=which_waveform)
    
    # Plot the comparison
    corner_kwargs = default_corner_kwargs
    corner_kwargs["color"] = my_colors["jim"]
    fig = corner.corner(peter_samples, labels = LABELS, hist_kwargs={'density': True}, **default_corner_kwargs)
    corner_kwargs["color"] = my_colors["bilby"]
    corner.corner(gwosc_samples, fig = fig, labels = LABELS, hist_kwargs={'density': True}, **default_corner_kwargs)
    
    # Remove the t_c label for comparison with bilby
    if remove_tc:
        labels = copy.deepcopy(LABELS)
        labels.remove(r'$t_c$')
    else:
        labels = LABELS
        
    # TODO improve the plot, e.g. give a custom legend
    save_name += "_bilby"
    print(f"Saving to {save_name}")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{save_name}.{ext}", bbox_inches='tight')
    

def compute_js_divergences(jim_chains: jnp.array, 
                           bilby_chains: jnp.array,
                           plot_name: str):
    
    js_dict = {}
    
    idx_counter = 0
    for parameter_name in jim_naming:
        if parameter_name == "t_c":
            continue
        values_jim = jim_chains[:, idx_counter]
        values_bilby = bilby_chains[:, idx_counter]
        
        # Create histograms to then compute JS divergence
        histogram_jim, edges = np.histogram(values_jim, bins=20, density=True)
        histogram_bilby, _ = np.histogram(values_bilby, bins=edges, density=True)
        
        js_div = jensenshannon(histogram_jim, histogram_bilby, base = 2) ** 2
        js_dict[parameter_name] = js_div
        
        plt.figure(figsize=(10, 6))
        plt.stairs(histogram_jim, edges, label="jim", color = "blue")
        plt.stairs(histogram_bilby, edges, label="bilby", color = "red")
        plt.legend()
        plt.savefig(f"./figures/{plot_name}_js_div_{parameter_name}.png")
        plt.close()
        
        idx_counter += 1
        
    # Save it with pickle
    filename = f"../figures/{plot_name}_js_div.pkl"
    print(f"Saving the JS divergences to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(js_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return js_dict

############
### MAIN ###
############

def compare_jim_pbilby():
    
    start_time = time.time()
    save_path = "../figures/"
    convert_chi = True
    convert_lambdas = True
    
    events_to_plot = [
        "GW170817_NRTidalv2",
        "GW170817_TaylorF2",
        "GW190425_TaylorF2",
        "GW190425_NRTidalv2",
]
    
    for event in events_to_plot:
        paths = paths_dict[event]
        
        print("==============================================")
        print(f"Comparing runs for: {event}")
        print("==============================================")
        jim_path = paths["jim"]
        bilby_path = paths["bilby"]
        
        print("jim_path")
        print(jim_path)
        
        print("bilby_path")
        print(bilby_path)
        
        if "TaylorF2" in jim_path:
            which_waveform = "TaylorF2"
        else:
            which_waveform = "PhenomDNRT"
        
        corner_kwargs = copy.deepcopy(default_corner_kwargs)
        
        # Fetch the desired kwargs from the specified dict
        # range = utils_compare_runs.get_ranges(event, convert_chi, convert_lambdas)
        range = None
        corner_kwargs["range"] = range
        idx_list = utils_compare_runs.get_idx_list(event, convert_chi = convert_chi, convert_lambdas = convert_chi)
        
        print("Reading bilby data")
        if ".h5" in bilby_path:
            bilby_samples = utils_compare_runs.get_chains_GWOSC(bilby_path, which_waveform=which_waveform)
        else:
            print(f"bilby_path: {bilby_path}")
            bilby_samples = utils_compare_runs.get_chains_bilby(bilby_path)

        print("Reading jim data")
        print(f"jim_path: {jim_path}")
        jim_samples = utils_compare_runs.get_chains_jim(jim_path, remove_tc = True)
        print("Loading data complete")
        
        save_name = save_path + event
        if "no_taper" in jim_path:
            save_name += "_no_taper"
        
        print("Starting plotting...")
        plot_comparison(jim_samples, 
                        bilby_samples, 
                        idx_list,
                        use_weights = False,
                        save_name = save_name,
                        which_waveform = which_waveform,
                        remove_tc = True,
                        convert_chi = convert_chi,
                        convert_lambdas = convert_lambdas,
                        corner_kwargs=corner_kwargs)
        
        print("Starting plotting... done!")
        # ====== Computing the JS divergences ======
        
        jim_chains = utils_compare_runs.get_chains_jim(jim_path)
        bilby_chains = utils_compare_runs.get_chains_bilby(bilby_path)
        
        js_dict = compute_js_divergences(jim_chains,
                                         bilby_chains, 
                                         plot_name = event)
        
        print(js_dict)
        
        
    print("DONE")
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")
        

def compare_taper_runs():
    taper_path = "/home/thibeau.wouters/TurboPE-BNS/real_events/"
    no_taper_path = "/home/thibeau.wouters/TurboPE-BNS/real_events_no_taper/"
    
    for run_name in ["GW170817_NRTidalv2", "GW190425_NRTidalv2"]:
    
        taper_filename = os.path.join(taper_path, f"{run_name}/outdir/results_production.npz")
        if run_name == "GW170817_NRTidalv2":
            no_taper_filename = "/home/thibeau.wouters/TurboPE-BNS/real_events_no_taper/GW170817_NRTidalv2/outdir_965341/results_production.npz"
        else:
            no_taper_filename = os.path.join(no_taper_path, f"{run_name}/outdir/results_production.npz")
        
        print(f"Making plots for no_taper_filename: {no_taper_filename}")

        taper_samples = utils_compare_runs.get_chains_jim(taper_filename)
        no_taper_samples = utils_compare_runs.get_chains_jim(no_taper_filename)
        
        convert_chi = True
        convert_lambdas = True
        
        # TODO: typo here? Breaks the code
        # n_dim = 13
        # if convert_chi:
        #     n_dim -= 1
        # if convert_lambdas:
        #     n_dim -= 1
        
        idx_list = [0] * 11
        
        # range = utils_compare_runs.get_ranges(run_name, convert_chi=convert_chi, convert_lambdas=convert_lambdas)
        range = None
        
        corner_kwargs = copy.deepcopy(default_corner_kwargs)
        corner_kwargs["range"] = range
        
        kwargs = {"first_color": my_colors["jim_no_taper"],
                "second_color": my_colors["jim"],
                "first_label": r"\textsc{Jim} (w/o taper)",
                "second_label": r"\textsc{Jim} (w/ taper)",
                "histogram_fill_color": "#b7d8e9"
        }
        
        save_name = f"../figures/taper_comparison_{run_name}"
        
        js_div = compute_js_divergences(taper_samples, no_taper_samples, plot_name = "")
        
        plot_comparison(taper_samples, 
                        no_taper_samples,
                        idx_list = idx_list,
                        use_weights = False,
                        save_name = save_name,
                        which_waveform = "NRTidalv2",
                        convert_chi=convert_chi,
                        convert_lambdas=convert_lambdas,
                        corner_kwargs = corner_kwargs,
                        **kwargs
                        )
        
        print("DONE")
    
def main():
    compare_jim_pbilby()
    # compare_taper_runs()
    
    # compute_js_divergences() # TODO: remove?
        
if __name__ == "__main__":
    main()