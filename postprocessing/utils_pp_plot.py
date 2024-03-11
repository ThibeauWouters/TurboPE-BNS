import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import copy
import numpy as np
import json
import arviz 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import kstest, uniform, percentileofscore
import jax.numpy as jnp

from ripple import get_chi_eff, Mc_eta_to_ms, lambdas_to_lambda_tildes, lambda_tildes_to_lambdas
from arviz import hdi
from scipy.optimize import bisect, root_scalar

import sys
sys.path.append("../tidal/")

from tqdm import tqdm
from typing import Callable

### Hyperparameters
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

red = (255 / 260, 59 / 260, 48 / 260)
blue = (0 / 260, 122 / 260, 255 / 260)
orange = (255 / 260, 149 / 260, 0 / 260)
brown = (220 / 260, 114 / 260, 21 / 260)
yellow = (255 / 260, 204 / 260, 0 / 260)
green = (52 / 260, 199 / 260, 89 / 260) # this is: original version
green = (36 / 260, 138 / 260, 61 / 260) # this is: accessible light
teal = (48 / 260, 176 / 260, 199 / 260)
indigo = (54 / 260, 52 / 260, 163 / 260)
purple = (175 / 260, 82 / 260, 222 / 260)

# my_colors = [red, blue, orange, indigo, green, teal, purple]
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#e41a1c', '#dede00'] # '#999999', 

my_colors = CB_color_cycle

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=24),
                        title_kwargs=dict(fontsize=24), 
                        color="blue",
                        levels=[0.68, 0.95, 0.997],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

labels_tidal_lambda12 = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

labels_tidal_lambda12_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

labels_tidal_deltalambda = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

labels_tidal_deltalambda_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

# This is our default
labels_tidal = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

# For the combined, these are the values:
p_calculation_dict_NRTv2 = {"M_c": "one_sided",
                      "q": "one_sided",
                      "s1_z": "two_sided",
                      "s2_z": "two_sided",
                      "lambda_1": "one_sided", # for lambda tilde: one_sided, for lambdas: two_sided
                      "lambda_2": "two_sided", # for both: two_sided is best
                      "d_L": "two_sided",
                      "t_c": "one_sided",
                      "phase_c": "two_sided", # "one_sided"
                      "cos_iota": "one_sided",
                      "psi": "one_sided", # "one_sided"
                      "ra": "one_sided", # "two_sided"
                      "sin_dec": "one_sided",
                      "chi_eff": "two_sided"
                      }

p_calculation_dict_TF2 = {"M_c": "one_sided",
                      "q": "one_sided",
                      "s1_z": "two_sided",
                      "s2_z": "two_sided",
                      "lambda_1": "one_sided", # for lambda tilde: one_sided, for lambdas: two_sided
                      "lambda_2": "two_sided", # for both: two_sided
                      "d_L": "two_sided",
                      "t_c": "one_sided",
                      "phase_c": "circular", # "one_sided"
                      "cos_iota": "one_sided",
                      "psi": "one_sided", # "one_sided"
                      "ra": "two_sided", # "two_sided"
                      "sin_dec": "two_sided",
                      "chi_eff": "two_sided"
                      }

# p_calculation_dict_values_TF2 = list(p_calculation_dict_TF2.values())
# p_calculation_dict_values_NRTv2 = list(p_calculation_dict_NRTv2.values())

p_calculation_dict_wf = {"TF2": p_calculation_dict_TF2,
                        "NRTv2": p_calculation_dict_NRTv2
}


PRIOR = {
        "M_c": [0.8759659737275101, 2.6060030916165484],
        "q": [0.5, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_1": [0.0, 5000.0], 
        "lambda_2": [0.0, 5000.0], 
        "d_L": [30.0, 300.0], 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 2 * jnp.pi], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, jnp.pi], 
        "ra": [0.0, 2 * jnp.pi], 
        "sin_dec": [-1, 1]
}
NAMING = list(PRIOR.keys())

#########################
### PP-PLOT UTILITIES ###
#########################


def get_credible_level_scipy(samples: np.array, 
                             injected_value: float, 
                             idx: int):
    """
    This function computes the credible level using the scipy functionalities.

    Args:
        samples (np.array): Posterior samples
        injected_value (float): _description_
        circular (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    circular = False
    print("idx")
    print(idx)
    
    name = list(PRIOR.keys())[idx]
    print("name")
    print(name)
    if idx in [8, 10, 11]: # phic, psi, alpha
        circular = True
        
        original_samples = copy.deepcopy(samples)
        
        # Do appropriate changes
        if idx == 8: # phic, is below pi/2
            samples = 4 * (samples % (np.pi / 2))
            # check if all close with original
            print("DEBUG: allclose for phase_c?")
            print(np.allclose(samples, original_samples))
            
            injected_value = 4 * (injected_value % (np.pi / 2))
            
        if idx == 10: # psi, is below pi
            samples = 4 * (samples % (np.pi / 2))
            
            injected_value = 4 * (injected_value % (np.pi / 2))
        
        # Map the samples to the [-pi, pi] range for circular parameters
        samples = samples - np.pi
        injected_value = injected_value - np.pi
    
    print("samples")
    print(samples)
    
    print("injected_value")
    print(injected_value)
    
    if circular:
        # check if the samples and the injected value is within [-pi, pi]
        if np.any(samples > np.pi) or np.any(samples < -np.pi):
            raise ValueError("Samples outside of the [-pi, pi] range")
        if injected_value > np.pi or injected_value < -np.pi:
            raise ValueError("Injected value outside of the [-pi, pi] range")

    def f(p):
        left, right = hdi(samples, hdi_prob=p, circular=circular)
        dist_left = injected_value - left
        dist_right = injected_value - right
        if np.abs(dist_left) > np.abs(dist_right):
            return dist_right
        else:
            return dist_left

    eps = 1e-3
    first_p = eps
    last_p = 1 - eps
    
    p_array = np.linspace(first_p, last_p, 100)
    f_array = np.array([f(p) for p in p_array])
    print("f_array")
    print(f_array)
    
    ## Old code
    return bisect(f, first_p, last_p, full_output=False)
    
    # ## New code (different method)
    # method = 'newton'
    # root = root_scalar(f, method=method, x0 = 0.5)
    
    print("root")
    print(root)
    
    return root

def get_credible_level_circular(samples: np.array, 
                                injected_value: float, 
                                name: str,
                                nb_bins: int = 20,
                                debug_plotting: bool = False):
    """
    This function computes the credible level using the scipy functionalities.

    Args:
        samples (np.array): Posterior samples
        injected_value (float): _description_
        circular (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    # Do appropriate changes
    if name in ["phase_c", "psi"]:
        samples = 4 * (samples % (np.pi / 2))
        injected_value = 4 * (injected_value % (np.pi / 2))
        
    # Get the histogram for nb_bins bins, then find the location of the largest bin
    hist, bin_edges = np.histogram(samples, bins = nb_bins, range=(0, 2 * np.pi))
    max_bin = np.argmax(hist)
    # Get the middle of the bin edge with highest count
    bin_spacing = bin_edges[1] - bin_edges[0]
    middle = bin_edges[max_bin] + bin_spacing / 2
    old_injected_value = injected_value # for plotting
    
    # Shift all the samples and injected value by the middle, so that middle is at zero now
    samples = samples - middle + np.pi
    injected_value = injected_value - middle + np.pi
    
    # Make sure they are mapped to the [0, 2pi] range
    samples = samples % (2 * np.pi)
    injected_value = injected_value % (2 * np.pi)
    
    if debug_plotting:
        new_hist, new_bin_edges = np.histogram(samples, bins = 50, range=(0, 2 * np.pi))
        # make a plot to check
        plt.stairs(hist, bin_edges, color="blue", linewidth=2)
        plt.axvline(old_injected_value, color="blue", linestyle="-", label="old inj")
        plt.axvline(middle, color="black", linestyle="--", label="middle")
        plt.stairs(new_hist, new_bin_edges, color="red", linewidth=2)
        plt.axvline(injected_value, color="red", linestyle="-", label="new inj")
        # plt.legend()
        plt.savefig(f"./pp_TaylorF2/histograms/circular_{name}_{injected_value}.png")
        plt.close()
        
    # Get the percentile of the injected value
    percentile = percentileofscore(samples, injected_value) / 100
    
    # TODO toggle one sided or two sided by the user
    percentile = 1 - 2 * min(percentile, 1-percentile)
    
    return percentile
    

def get_mirror_location(samples: np.array) -> tuple[np.array, np.array]:
    """Computes the mirrored location of the samples in the sky.

    Args:
        samples (np.array): Posterior values or true values

    Returns:
        tuple[np.array, np.array]: The mirrored samples in the sky, two versions for two signs of t_c.
    """
    
    # Just to be sure, make a deepcopy
    mirror_samples = copy.deepcopy(samples)
    
    # Check whether we have a list of samples, or a single sample
    if len(np.shape(mirror_samples)) == 1:
        mirror_samples = np.array([mirror_samples])
        
    # Get the parameter names
    naming = list(PRIOR.keys())
    
    # Get indices of parameters for which we will perform a transformation
    alpha_index = naming.index("ra")
    delta_index = naming.index("sin_dec")
    iota_index  = naming.index("cos_iota")
    phi_c_index = naming.index("phase_c")
    t_c_index   = naming.index("t_c")
    
    # First transform iota and delta
    mirror_samples[:, iota_index] = np.arccos(mirror_samples[:, iota_index])
    mirror_samples[:, delta_index] = np.arcsin(mirror_samples[:, delta_index])
    
    # Do the transformations:
    mirror_samples[:, alpha_index] = (mirror_samples[:, alpha_index] + np.pi) % (2 * np.pi)
    mirror_samples[:, delta_index] =  - mirror_samples[:, delta_index]
    mirror_samples[:, iota_index]  =  np.pi - mirror_samples[:, iota_index]
    mirror_samples[:, phi_c_index] = (mirror_samples[:, phi_c_index] + np.pi) % (2 * np.pi)
    
    ## TODO check the t_c transformation
    R_e = 6.378e+6
    c   = 299792458
    # Will have on with plus, and one with minus, so copy whatever we have already now
    second_mirror_samples = copy.deepcopy(mirror_samples)
    # Also will return a copy where t_c was not changed:
    mirror_samples_same_tc = copy.deepcopy(mirror_samples)
    mirror_samples[:, t_c_index] = mirror_samples[:, t_c_index] - R_e / c
    second_mirror_samples[:, t_c_index] = second_mirror_samples[:, t_c_index] + R_e / c
    
    # Convert iota and delta back to cos and sin values
    mirror_samples[:, iota_index] = np.cos(mirror_samples[:, iota_index])
    mirror_samples[:, delta_index] = np.sin(mirror_samples[:, delta_index])
    second_mirror_samples[:, iota_index] = np.cos(second_mirror_samples[:, iota_index])
    second_mirror_samples[:, delta_index] = np.sin(second_mirror_samples[:, delta_index])
    
    return mirror_samples_same_tc, mirror_samples, second_mirror_samples

def make_cumulative_histogram(data: np.array, nb_bins: int = 100):
    """
    Creates the cumulative histogram for a given dataset.

    Args:
        data (np.array): Given dataset to be used.
        nb_bins (int, optional): Number of bins for the histogram. Defaults to 100.

    Returns:
        np.array: The cumulative histogram, in density.
    """
    h = np.histogram(data, bins = nb_bins, range=(0,1), density=True)
    return np.cumsum(h[0]) / np.sum(h[0])

def make_uniform_cumulative_histogram(size: tuple, nb_bins: int = 100) -> np.array:
    """
    Generates a cumulative histogram from uniform samples.
    
    Size: (N, dim): Number of samples from the uniform distribution, counts: nb of samples

    Args:
        counts (int): Dimensionality to be generated.
        N (int, optional): Number of samples from the uniform distribution. Defaults to 10000.

    Returns:
        np.array: Cumulative histograms for uniform distributions. Shape is (N, nb_bins)
    """
    
    uniform_data = np.random.uniform(size = size)
    cum_hist = []
    for data in uniform_data:
        result = make_cumulative_histogram(data, nb_bins = nb_bins)
        cum_hist.append(result)
        
    cum_hist = np.array(cum_hist)
    return cum_hist

def get_true_params_and_credible_level(chains: np.array, 
                                       true_params_list: np.array,
                                       return_first: bool = False,
                                       weight_fn: Callable = lambda x: x,
                                       which_percentile_calculation="one_sided",
                                       convert_to_chi_eff: bool=False,
                                       convert_to_lambda_tilde: bool=False,
                                       wf_name: str = "NRTv2"
                                       ) -> tuple[np.array, float]:
    """
    Finds the true parameter set from a list of true parameter sets, and also computes its credible level.

    Args:
        true_params_list (np.array): List of true parameters and copies for sky location.

    Returns:
        tuple[np.array, float]: The select true parameter set, and its credible level.
    """
    
    # Indices which have to be treated as circular
    naming = list(PRIOR.keys())
    
    p_calculation_dict_values = p_calculation_dict_wf[wf_name]
    
    # Build the weighting function
    d_L_index = naming.index("d_L")
    d_values = chains[:, d_L_index]
    d_values = np.array(d_values)
    # Compute the weights
    weights = weight_fn(d_values)
    weights = 1 / weights
    # Normalize the weights
    weights /= np.sum(weights)
    # Resample the chains based on these weights
    indices = np.random.choice(np.arange(len(chains)), size=len(chains), p=weights)
    chains = chains[indices]
    
    if return_first:
        # Ignore the sky location mirrors, just take the first one
        true_params = true_params_list[0]
        true_params_list = [true_params]
        
    if convert_to_chi_eff:
        # Convert to chi_eff
        mc_index = naming.index("M_c")
        q_index = naming.index("q")
        chi1_index = naming.index("s1_z")
        chi2_index = naming.index("s2_z")
        for i, true_param in enumerate(true_params_list):
            
            chi_eff_true = compute_chi_eff(true_param[mc_index], true_param[q_index], true_param[chi1_index], true_param[chi2_index])
            new_true_params = np.array([true_param[mc_index], true_param[q_index], chi_eff_true, true_param[4], true_param[5], true_param[6], true_param[7], true_params[8], true_param[9], true_param[10], true_param[11], true_param[12]])
            true_params_list[i] = new_true_params
            
        # Also convert the chains
        chi_eff = compute_chi_eff(chains[:, mc_index], chains[:, q_index], chains[:, chi1_index], chains[:, chi2_index])
        chains = np.array([chains[:, mc_index], chains[:, q_index], chi_eff, chains[:, 4], chains[:, 5], chains[:, 6], chains[:, 7], chains[:, 8], chains[:, 9], chains[:, 10], chains[:, 11], chains[:, 12]]).T
        
        # Delete chi2 from naming
        naming.pop(chi2_index)
        
    if convert_to_lambda_tilde:
        lambda1_index = naming.index("lambda_1")
        lambda2_index = naming.index("lambda_2")
        
        for i, true_param in enumerate(true_params_list):
            mc, q = true_param[0], true_param[1]
            eta = q/(1+q)**2
            lambda1 = true_param[lambda1_index]
            lambda2 = true_param[lambda2_index]
            # Convert to component masses
            m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
            lambda_tilde_true, delta_lambda_tilde_true = lambdas_to_lambda_tildes(jnp.array([lambda1, lambda2, m1, m2]))
            
            # Replace lambdas with new lambdas
            new_true_params = copy.deepcopy(true_param)
            new_true_params[lambda1_index] = lambda_tilde_true
            new_true_params[lambda2_index] = delta_lambda_tilde_true
            
            # Save back in the list
            true_params_list[i] = new_true_params
            
        # Also convert the chains
        mc = chains[:, mc_index]
        q = chains[:, q_index]
        eta = q/(1+q)**2
        lambda1 = chains[:, lambda1_index]
        lambda2 = chains[:, lambda2_index]
        m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
        lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([lambda1, lambda2, m1, m2]))
        chains[:, lambda1_index] = lambda_tilde
        chains[:, lambda2_index] = delta_lambda_tilde
    
    # When checking sky reflected as well, iterate over all "copies"
    supported_which_percentile_calculation = ["one_sided", "two_sided", "scipy", "combined", "circular"]
    if which_percentile_calculation not in supported_which_percentile_calculation:
        print(f"ERROR: which_percentile_calculation is not supported. Supported are: {supported_which_percentile_calculation}")
        print("Changing to one_sided")
        which_percentile_calculation = "one_sided"
        
    for i, true_params in enumerate(true_params_list):
        params_credible_level_list = []
        
        # Iterate over each parameter of this "copy" of parameters
        for j, param in enumerate(true_params):
            
            # Fetch the name and its calculation key if desired
            this_param_name = naming[j]
            if which_percentile_calculation == "combined":
                local_which_percentile_calculation = p_calculation_dict_values[this_param_name]
                # print("this_param_name")
                # print(this_param_name)
                # print("local_which_percentile_calculation")
                # print(local_which_percentile_calculation)
            else:
                local_which_percentile_calculation = which_percentile_calculation
            
            # Get the q value in case it is needed below
            q = percentileofscore(chains[:, j], param) / 100
            
            # TODO add the combined option here
            if local_which_percentile_calculation == "one_sided":
                credible_level = q
            elif local_which_percentile_calculation == "two_sided":
                credible_level = 1 - 2 * min(q, 1-q)
            # NOTE scipy is broken!
            elif local_which_percentile_calculation == "scipy":
                credible_level = get_credible_level_scipy(chains[:, j], param, idx = j)
            elif local_which_percentile_calculation == "circular":
                name = naming[j]
                credible_level = get_credible_level_circular(chains[:, j], param, name = name)
                
            params_credible_level_list.append(credible_level)
        
        params_credible_level_list = np.array(params_credible_level_list)
        
        if i == 0:
            credible_level_list = params_credible_level_list 
        else:
            credible_level_list = np.vstack((credible_level_list, params_credible_level_list))
            
        # Now choose the correct index
        if return_first:
            credible_level_list = np.reshape(credible_level_list, (1, -1))
        summed_credible_level = np.sum(abs(0.5 - credible_level_list), axis = 1)
        # Pick the index with the lowest sum
        idx = np.argmin(summed_credible_level)
        true_params = true_params_list[idx]
        credible_level = credible_level_list[idx]
    
    return true_params, credible_level

def get_credible_levels_injections(outdir: str, 
                                   return_first: bool = True,
                                   max_number_injections: int = -1,
                                   weight_fn: Callable = lambda x: x,
                                   thinning_factor: int = 100,
                                   which_percentile_calculation: str = "one_sided",
                                   save: bool = True,
                                   convert_cos_sin: bool = True,
                                   convert_to_chi_eff: bool = False,
                                   convert_to_lambda_tilde: bool = False,
                                   wf_name: str = "NRTv2") -> np.array:
    """
    Compute the credible levels list for all the injections. 
    
    Args:
        reweigh_distance (bool, optional): Whether to reweigh based on the distance or not. Defaults to False.
        return_first (bool, optional): Whether to return the first true parameter set (don't take sky location mirrors into account) or not. Defaults to False.

    Returns:
        np.array: Array of credible levels for each injection.
    """
    
    # Get parameter names
    naming = list(PRIOR.keys())
        
    print("naming")
    print(naming)
    print("wf_name")
    print(wf_name)
    n_dim = len(naming)
    
    print("Reading injection results")
    
    credible_level_list = []
    subdirs = []
    counter = 0
    
    print("Iterating over the injections, going to compute the credible levels")
    # print("NOTE: reweigh_distance is set to ", reweigh_distance)
    for subdir in tqdm(os.listdir(outdir)):
        subdir_path = os.path.join(outdir, subdir)
        
        if os.path.isdir(subdir_path):
            json_path = os.path.join(subdir_path, "config.json")
            chains_filename = f"{subdir_path}/results_production.npz"
            if not os.path.isfile(json_path) or not os.path.isfile(chains_filename):
                continue
            
            subdirs.append(subdir)
            
            # Load config, and get the injected parameters
            with open(json_path, "r") as f:
                config = json.load(f)
            true_params = np.array([config[name] for name in naming])
                
            # Get the recovered parameters
            data = np.load(chains_filename)
            chains = data['chains'].reshape(-1, n_dim)
            
            # Thin a bit to reduce computational complexity a bit
            chains = chains[::thinning_factor]
            
            # Convert true params and samples for cos_iota and sin_dec
            if convert_cos_sin:
                cos_iota_index = naming.index("cos_iota")
                sin_dec_index = naming.index("sin_dec")
                true_params[cos_iota_index] = np.arccos(true_params[cos_iota_index])
                true_params[sin_dec_index] = np.arcsin(true_params[sin_dec_index])
                chains[:, cos_iota_index] = np.arccos(chains[:, cos_iota_index])
                chains[:, sin_dec_index] = np.arcsin(chains[:, sin_dec_index])
            
            # Get the sky mirrored values as well, NOTE this outputs an array of arrays!
            if not return_first:
                mirrored_values = get_mirror_location(true_params) # tuple
                mirrored_values = np.vstack(mirrored_values) # np array
                all_true_params = np.vstack((true_params, mirrored_values))
            else:
                all_true_params = np.array([true_params])
            
            
            true_params, credible_level = get_true_params_and_credible_level(chains, 
                                                                             all_true_params, 
                                                                             return_first=return_first,
                                                                             weight_fn=weight_fn,
                                                                             which_percentile_calculation=which_percentile_calculation,
                                                                             convert_to_chi_eff=convert_to_chi_eff,
                                                                             convert_to_lambda_tilde=convert_to_lambda_tilde,
                                                                             wf_name=wf_name)
            
            credible_level_list.append(credible_level)
            
            # Also save the credible level
            if save:
                filename = f"{subdir_path}/credible_level.npz"
                np.savez(filename, credible_level=credible_level)
            
            counter += 1
            if counter == max_number_injections:
                print(f"INFO: Stopping after {max_number_injections} injections.")
                break
            
    credible_level_list = np.array(credible_level_list)
    
    return credible_level_list, subdirs

def compute_chi_eff(mc, q, chi1, chi2):
    eta = q/(1+q)**2
    m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
    params = jnp.array([m1, m2, chi1, chi2])
    chi_eff = get_chi_eff(params)
    
    return chi_eff

#################
### DEBUGGING ###
#################

def plot_distributions_injections(outdir: str, 
                                  param_index: int = 0,
                                  **plotkwargs) -> None:
    """
    TODO
    
    By default, we are checking chirp mass
    """
    naming = list(PRIOR.keys())
    param_name = naming[param_index]
    print("Checking parameter: ", param_name)
    
    plt.figure(figsize=(12, 9))
    print("Iterating over the injections, going to compute the credible levels")
    for subdir in tqdm(os.listdir(outdir)):
        subdir_path = os.path.join(outdir, subdir)
        
        if os.path.isdir(subdir_path):
            json_path = os.path.join(subdir_path, "config.json")
            chains_filename = f"{subdir_path}/results_production.npz"
            if not os.path.isfile(json_path) or not os.path.isfile(chains_filename):
                continue
            
            # Load config, and get the true (injected) parameters
            with open(json_path, "r") as f:
                config = json.load(f)
            true_params = np.array([config[name] for name in naming])
            
            true_param = true_params[param_index]
            
            # Get distribution of samples
            data = np.load(chains_filename)
            chains = data['chains'].reshape(-1, len(naming))
            samples = chains[:, param_index]
            
            samples -= true_param
            # Make histogram
            plt.hist(samples, **plotkwargs)
            
    plt.axvline(0, color = "black", linewidth = 1)
    plt.xlabel(f"Residuals of {param_name}")
    plt.ylabel("Density")
    plt.savefig(f"./pp_TaylorF2/distributions_{param_name}.png")
    plt.close()
            
            
def plot_credible_levels_injections(outdir: str,
                                    param_index: int = 0
) -> None:
    """
    TODO
    """
    naming = list(PRIOR.keys())
    param_name = naming[param_index]
    print("Checking parameter: ", param_name)
    
    credible_level_list, _ = get_credible_levels_injections(outdir)
    
    plt.figure(figsize=(12, 9))
    plt.hist(credible_level_list[:, param_index], bins = 20, histtype="step", color="blue", linewidth=2, density=True)
    plt.xlabel("Credible level")
    plt.ylabel("Density")
    plt.title(f"Credible levels {param_name}")
    plt.savefig(f"./pp_TaylorF2/credible_levels_{param_name}.png")
    plt.close()


### TODO remove this? Deprecated

# def analyze_credible_levels(credible_levels, 
#                             subdirs, 
#                             param_index = 0, 
#                             nb_round: int = 5):
#     subdirs = np.array(subdirs)
    
#     credible_levels_param = credible_levels[:, param_index]
    
#     # Sort 
#     sorted_indices = np.argsort(credible_levels_param)
#     credible_levels_param = credible_levels_param[sorted_indices]
#     credible_levels = credible_levels[sorted_indices]
#     subdirs_sorted = subdirs[sorted_indices]
    
#     for i, (subdir, credible_level) in enumerate(zip(subdirs_sorted, credible_levels_param)):
#         print(f"{subdir}: {np.round(credible_level, nb_round)}")

# ####################################
# ### Further postprocessing tools ###
# ####################################

# def test_tile():
#     array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
#     print("np.shape(array)")
#     print(np.shape(array))

#     # Repeat along the second axis to create the desired shape (1000, 5, 13)
#     result_array = np.tile(array[:, np.newaxis, :], (1, 5, 1))

#     print("np.shape(result_array)")
#     print(np.shape(result_array))

#     print(result_array[:, 0, :])
    
def my_format(number: float):
    return "{:.2f}".format(number)
    
def analyze_runtimes(outdir, verbose: bool = True):
    runtimes = []
    for dir in os.listdir(outdir):
        runtime_file = outdir + dir + "/runtime.txt"
        if not os.path.exists(runtime_file):
            continue
        runtime = np.loadtxt(runtime_file)
        runtimes.append(runtime)
        
    runtimes = np.array(runtimes)
    if verbose:
        print(f"Mean +- std runtime: {my_format(np.mean(runtimes))} +- {my_format(np.std(runtimes))} seconds")
        print(f"Min runtime: {my_format(np.min(runtimes))} seconds")
        print(f"Max runtime: {my_format(np.max(runtimes))} seconds")
        print(f"Median runtime: {my_format(np.median(runtimes))} seconds")

        print("\n\n")
        runtimes /= 60
        print(f"Mean +- std runtime: {my_format(np.mean(runtimes))} +- {my_format(np.std(runtimes))} minutes")
        print(f"Min runtime: {my_format(np.min(runtimes))} minutes")
        print(f"Max runtime: {my_format(np.max(runtimes))} minutes")
        print(f"Median runtime: {my_format(np.median(runtimes))} minutes")

        print("\n\n")
        print(f"Number of runs: {len(runtimes)}")
    
    return runtimes