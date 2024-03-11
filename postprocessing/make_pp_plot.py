import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
os.environ["CUDA_VISIBILE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
import numpy as np
# import arviz 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import kstest, uniform, percentileofscore

import utils_pp_plot as utils_pp_plot
import importlib
importlib.reload(utils_pp_plot)
plt.rcParams.update(utils_pp_plot.matplotlib_params)

###############################
### Post-injection analysis ###
###############################

naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

def make_pp_plot(credible_level_list: np.array, 
                 which_percentile_calculation,
                 percentile_list: list = [0.68, 0.95, 0.995], 
                 nb_bins: int = 300,
                 params_idx: list = None,
                 reweigh_distance: bool = False,
                 convert_to_chi_eff: bool = False,
                 convert_to_lambda_tilde: bool = False,
                 labels = utils_pp_plot.labels_tidal,
                 save_name: str = "pp_plot"
                 ) -> None:
    """
    Creates a pp plot from the credible levels.

    Args:
        credible_level_list (np.array): List of crxedible levels obtained from injections.
        percentile (float/list, optional): Percentile used for upper and lower quantile. Defaults to 0.05.
        nb_bins (int, optional): Number of bins in the histogram. Defaults to 100.
    """
    
    # Group the plotting hyperparameters here: 
    bbox_to_anchor = (1.025, 1.0)
    legend_fontsize = 26
    handlelength = 1
    linewidth = 2
    min_alpha = 0.15
    max_alpha = 0.25
    shadow_color = "gray"
    n = np.shape(credible_level_list)[1]
    color_list = cm.rainbow(np.linspace(0, 1, n))
    
    # First, get uniform distribution cumulative histogram:
    nb_injections, n_dim = np.shape(credible_level_list)
    print("nb_injections, n_dim")
    print(nb_injections, n_dim)
    N = 10_000
    uniform_histogram = utils_pp_plot.make_uniform_cumulative_histogram((N, nb_injections), nb_bins=nb_bins)
    
    # Check if given percentiles is float or list
    if isinstance(percentile_list, float):
        percentile_list = [percentile_list]
    # Convert the percentages
    percentile_list = [1 - p for p in percentile_list]
    # Get list of alpha values:
    alpha_list = np.linspace(min_alpha, max_alpha, len(percentile_list))
    alpha_list = alpha_list[::-1]
        
    plt.figure(figsize=(12, 9))    
    # Plot the shadow bands
    for percentile, alpha in zip(percentile_list, alpha_list):
        
        upper_quantile_array = []
        lower_quantile_array = []
        for i in range(nb_bins):
            upper_quantile_array.append(np.quantile(uniform_histogram[:, i], (1-percentile/2)))
            lower_quantile_array.append(np.quantile(uniform_histogram[:, i], (percentile/2)))
        
        bins = np.linspace(0, 1, nb_bins + 1)
        bins = (bins[1:]+bins[:-1])/2
        plt.fill_between(bins, lower_quantile_array, upper_quantile_array, color = shadow_color, alpha = alpha)
    
    # Compute the x data for the plot
    x = np.append(0, bins)
    # Will save the computed pvalues here
    pvalues = []
    print("Creating pp-plot, getting p values . . .")
    if params_idx is None:
        params_idx = range(n_dim)
        
    linestyles = ["-", "--"]
    for i in params_idx: 
        # col = color_list[i] ### TODO remove, old code
        
        # Find correct label and color and linestyle
        label = labels[i]
        col_index = i % len(utils_pp_plot.my_colors)
        ls_index = i // len(utils_pp_plot.my_colors)
        col = utils_pp_plot.my_colors[col_index]
        ls = linestyles[ls_index]
        
        # Compute the p-value
        p = kstest(credible_level_list[:nb_injections, i], cdf = uniform(0,1).cdf).pvalue
        
        # Compute the y data for the plot
        y = np.append(0, utils_pp_plot.make_cumulative_histogram(credible_level_list[:nb_injections, i], nb_bins=nb_bins))
        
        # Finally, plot
        plt.plot(x, y, c=col, ls=ls, label = f"{label} ($p = {p:.2f}$) ", linewidth = linewidth)
        pvalues.append(p)
    
    plt.legend(bbox_to_anchor = bbox_to_anchor, fontsize = legend_fontsize, handlelength=handlelength)
    plt.xlabel(r'Percentile $x$')
    plt.ylabel(r'Fraction of events with percentile $\leq x$')
    print("Creating pp-plot, getting p values . . . DONE")

    print("pvalues")
    print(pvalues)
    ptotal = kstest(pvalues, cdf=uniform(0,1).cdf).pvalue
    string_total = f"N = {len(credible_level_list)}, Total p-value: {ptotal:.2f}"
    print(string_total)
    
    # TODO add p value or not?
    title_string = r"$N = {}$".format(len(credible_level_list))
    title_string = ""
    print(title_string)
    
    print("Saving pp-plot")
    plt.grid(False) # disable grid
    plt.title(title_string)
    save_name += which_percentile_calculation
    if reweigh_distance:
        save_name += "_reweighed"
    if convert_to_chi_eff:
        save_name += "_chieff"
    if convert_to_lambda_tilde:
        save_name += "_lambdatilde"
        
    for ext in [".png", ".pdf"]:
        full_savename = save_name + ext
        print(f"Saving pp-plot to: {full_savename}")
        plt.savefig(full_savename, bbox_inches="tight")
        
#############
### Main ####
#############
        
if __name__ == "__main__":
    
    name = "NRTv2"
    outdir = f"../injections/outdir_{name}/"
    print(f"Making pp plot for outdir: {outdir}")
    figures_dir = f"../figures/"
    
    which_percentile_calculation = "combined" # which function to use to compute the percentiles
    convert_to_chi_eff = True # whether to convert spins to chi eff or not
    reweigh_distance = False # whether to reweigh samples based on the distance, due to the SNR cutoff used
    return_first = True # whether to just use injected params or to also look at sky mirrored locations
    convert_cos_sin = True # convert from cos iota and sin dec to iota and dec
    convert_to_lambda_tilde = False # whether to convert to lambda tildes or not
    thinning_factor = 1
    compute_percentiles = False
    
    print(f"script hyperparams: \nreweigh_distance = {reweigh_distance}, \nreturn_first = {return_first}, \nconvert_cos_sin = {convert_cos_sin}")
    
    if reweigh_distance:
        ## Deprectated
        # kde_file = "../postprocessing/kde_dL.npz"
        # data = np.load(kde_file)
        # kde_x = data["x"]
        # kde_y = data["y"]
        # weight_fn = lambda x: np.interp(x, kde_x, kde_y)
        pass
    else:
        weight_fn = lambda x: x
        
    # Choose the correct labels
    if convert_to_lambda_tilde:
        if convert_to_chi_eff:
            labels = utils_pp_plot.labels_tidal_deltalambda_chi_eff
        else:
            labels = utils_pp_plot.labels_tidal_deltalambda
    else:
        if convert_to_chi_eff:
            labels = utils_pp_plot.labels_tidal_lambda12_chi_eff
        else:
            labels = utils_pp_plot.labels_tidal_lambda12
        
    
    print(f"which_percentile_calculation is set to {which_percentile_calculation}")
    if compute_percentiles:
        credible_levels, subdirs = utils_pp_plot.get_credible_levels_injections(outdir, 
                                                                        return_first=return_first,
                                                                        weight_fn=weight_fn,
                                                                        thinning_factor=thinning_factor,
                                                                        which_percentile_calculation=which_percentile_calculation,
                                                                        save=False,
                                                                        convert_cos_sin=convert_cos_sin,
                                                                        convert_to_chi_eff=convert_to_chi_eff,
                                                                        convert_to_lambda_tilde=convert_to_lambda_tilde,
                                                                        wf_name=name)
    else:
        file = f"../figures/percentile_values_{name}_chieff.npz"
        print(f"Loading credible levels from {file}")
        data = np.load(file)
        credible_levels = data["credible_levels"]
    
    
    ### TF2 - checking which runs were potentially bad:
    # # ra index:
    # ra_idx = 11
    # # Sort the credible levels by ra:
    # sort_idx = np.argsort(credible_levels[:, ra_idx])
    # # Sort dirs
    # subdirs_sorted = np.array(subdirs)[sort_idx]
    # credible_levels_sorted = credible_levels[sort_idx]
    
    # for dir, cred in zip(subdirs_sorted, credible_levels_sorted):
    #     print(f"{dir} : {cred[ra_idx]}")
    
    # Limit to the first 100 injections for the plot:
    credible_levels = credible_levels[:100]
    
    make_pp_plot(credible_levels, 
                 which_percentile_calculation=which_percentile_calculation, 
                 reweigh_distance=reweigh_distance,
                 convert_to_chi_eff=convert_to_chi_eff,
                 convert_to_lambda_tilde=convert_to_lambda_tilde,
                 labels = labels,
                 save_name = f"../figures/pp_plot_{name}")
    
    # utils.analyze_runtimes(outdir)
    
    ### Also save the percentile values
    if compute_percentiles:
        print("Saving the percentile values")
        save_name = f"../figures/percentile_values_{name}"
        if reweigh_distance:
            save_name += "_reweighed"
        if convert_to_chi_eff:
            save_name += "_chieff"
        if convert_to_lambda_tilde:
            save_name += "_lambdatilde"
        save_name += ".npz"
        np.savez(save_name, credible_levels=credible_levels)
    
    print("DONE")