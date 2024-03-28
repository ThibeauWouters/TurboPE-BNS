import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import numpy as np
import json
from scipy.stats import percentileofscore, uniform, kstest
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ripple import get_chi_eff, Mc_eta_to_ms
from tqdm import tqdm

fs = 32
matplotlib_params = {"axes.grid": True,
          "text.usetex" : True,
          "font.family" : "serif",
          "ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "font.serif" : ["Times New Roman"],
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

# all injections are OK now
problematic_injections_dict = {"NRTv2": [],
                               "TF2": []}

naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
naming_chi_eff = ['M_c', 'q', 'chi_eff', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
labels_original = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']
labels_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin\delta$']

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#e41a1c', '#dede00']

my_colors = CB_color_cycle

def undo_periodicity(x):
    return 4 * (x % (np.pi / 2))

#################
### UTILITIES ###
#################

def compute_p_value_kstest(percentiles: dict):
    
    p_values = {}
    
    for key, value in percentiles.items():
        _, p_value = kstest(value, uniform.cdf)
        p_values[key] = p_value
        
    return p_values

def compute_chi_eff(samples):
    
    # Get the correct values
    mc = samples[naming.index("M_c")]
    q = samples[naming.index("q")]
    chi1 = samples[naming.index("s1_z")]
    chi2 = samples[naming.index("s2_z")]
        
    # Do conversions
    eta = q/(1+q)**2
    m1, m2 = Mc_eta_to_ms(np.array([mc, eta]))
    params = np.array([m1, m2, chi1, chi2])
    chi_eff = get_chi_eff(params)
    
    return chi_eff

def compute_p_value_kstest_dict(percentiles_dict: dict):
    
    p_values = {}
    for key, value in percentiles_dict.items():
        p_values[key] = compute_p_value_kstest(value)    
    return p_values

def postprocess_samples(samples, 
                        true_params,
                        convert_ra: bool = False,
                        convert_chi_eff: bool = True):
    
    
    phase_c_index = naming.index("phase_c")
    psi_index = naming.index("psi")
    
    for idx in [phase_c_index, psi_index]:
        samples[idx] = undo_periodicity(samples[idx])
        true_params[idx] = undo_periodicity(true_params[idx])
        
    if convert_ra:
        # TODO remove this this is just for a test
        ra_index = naming.index("ra")
        samples[ra_index] = undo_periodicity(samples[ra_index])
        true_params[ra_index] = undo_periodicity(true_params[ra_index])
        
        samples[ra_index] = np.sin(samples[ra_index])
        true_params[ra_index] = np.sin(true_params[ra_index])
    
    if convert_chi_eff:    
        # Compute chi eff values
        chi_eff_samples = compute_chi_eff(samples)
        chi_eff_true_params = compute_chi_eff(true_params)
        
        # Delete chi1 and chi2 from samples
        samples = np.delete(samples, [naming.index("s1_z"), naming.index("s2_z")], axis=0)
        true_params = np.delete(true_params, [naming.index("s1_z"), naming.index("s2_z")], axis=0)
        
        # Insert chi_eff
        samples = np.insert(samples, naming.index("s1_z"), chi_eff_samples, axis=0)
        true_params = np.insert(true_params, naming.index("s1_z"), chi_eff_true_params, axis=0)
        
    return samples, true_params

def get_credible_levels(wf_name: str,
                        apply_postprocess: bool = True,
                        convert_chi_eff: bool = True,
                        exclude_bad_injections: bool = True,
                        save: bool = True):
    
    # Fetch correct outdir
    outdir = outdirs_dict[wf_name]
    
    # This is a dict with keys being the methods for computing p values, and the values being dicts with names of parameters as keys and the percentiles as values
    credible_levels_dict = {"one_sided": {},
                            "two_sided": {}}
    
    if convert_chi_eff:
        local_naming = naming_chi_eff
    else:
        local_naming = naming
        
    for name in local_naming:
        for key in credible_levels_dict.keys():
            credible_levels_dict[key][name] = []
            
    # Iterate over the outdir
    for subdir in tqdm(os.listdir(outdir)):
        if not os.path.isdir(f"{outdir}/{subdir}"):
            continue
        if exclude_bad_injections:
            # Get subdir identifier
            try:
                number = int(subdir.split("_")[-1])
            except Exception as e:
                print("Could not get the number for dir:", subdir)
                print("Error: ", e)
                continue
            if number in problematic_injections_dict[wf_name]:
                continue
        # Check if results_production is there
        if "results_production.npz" in os.listdir(f"{outdir}/{subdir}"):
            # Loading data
            with open(f'{outdir}/{subdir}/config.json') as f:
                config = json.load(f)
                
            true_params = [config[name] for name in naming]
            
            all_posterior_samples = np.load(f'{outdir}/{subdir}/results_production.npz')
            all_posterior_samples = all_posterior_samples['chains'].T
            all_posterior_samples = [all_posterior_samples[param_idx].flatten() for param_idx in range(len(naming))]
            
            # Postprocess samples
            if apply_postprocess:
                all_posterior_samples, true_params = postprocess_samples(all_posterior_samples, true_params, convert_chi_eff=convert_chi_eff)
            
            # Iterate over parameters and compute percentiles
            for name, posterior_samples, true_param in zip(local_naming, all_posterior_samples, true_params):
                
                p_value = percentileofscore(posterior_samples, true_param) / 100.
                credible_levels_dict["one_sided"][name].append(p_value)
                two_sided = 1 - 2 * min(p_value, 1 - p_value)
                credible_levels_dict["two_sided"][name].append(two_sided)
        
    if save:
        # Save with pickle
        with open(f"./percentiles/credible_levels_dict_{wf_name}.pkl", "wb") as f:
            pickle.dump(credible_levels_dict, f)
        
    return credible_levels_dict

def choose_percentile_dict(percentile_dict):
    
    result = {} 
    p_values_dict = compute_p_value_kstest_dict(percentile_dict)
    
    all_param_names = list(p_values_dict.values())[0].keys()
    
    for param_name in all_param_names:
        # TODO what if more methods are used?
        if p_values_dict["one_sided"][param_name] > p_values_dict["two_sided"][param_name]:
            result[param_name] = percentile_dict["one_sided"][param_name]
        else:
            result[param_name] = percentile_dict["two_sided"][param_name]
        
    return result

###
### PLOTTING
###

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

def make_pp_plot(percentile_dict: dict, 
                 nb_bins: int = 500,
                 percentile_list: list = [0.68, 0.95, 0.995], 
                 save_name: str = "pp_plot"
                 ) -> None:
    """
    Creates a pp plot from the credible levels.
    """
    
    # Group the plotting hyperparameters here: 
    linewidth = 3
    min_alpha = 0.15
    max_alpha = 0.25
    shadow_color = "gray"
    
    n_dim = len(percentile_dict.keys())
    nb_injections = len(list(percentile_dict.values())[0])
    
    # First, get uniform distribution cumulative histogram:
    print("nb_injections, n_dim")
    print(nb_injections, n_dim)
    N = 10_000
    uniform_histogram = make_uniform_cumulative_histogram((N, nb_injections), nb_bins=nb_bins)
    
    # Check if given percentiles is float or list
    if isinstance(percentile_list, float):
        percentile_list = [percentile_list]
    # Convert the percentages
    percentile_list = [1 - p for p in percentile_list]
    # Get list of alpha values:
    alpha_list = np.linspace(min_alpha, max_alpha, len(percentile_list))
    alpha_list = alpha_list[::-1]
        
    plt.figure(figsize=(14, 10))    
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
    
    # Preparing to go finally to the pp plot
    print("Creating pp-plot . . .")
    
    pvalues = []
    linestyles = ["-", "--"]
    if n_dim == 13:
        labels = labels_original
    else:
        labels = labels_chi_eff
    
    # Hacky way to get the legends working:
    saved_labels = []
    saved_colors = []
    saved_linestyles = []
    for i, credible_level_list in enumerate(percentile_dict.values()): 
        
        # Find correct label and color and linestyle
        label = labels[i]
        col_index = i % len(my_colors)
        ls_index = i // len(my_colors)
        col = my_colors[col_index]
        ls = linestyles[ls_index]
        
        # Compute the p value
        _, p = kstest(credible_level_list, uniform.cdf)
        
        # Compute the y data for the plot
        y = np.append(0, make_cumulative_histogram(credible_level_list, nb_bins=nb_bins))
        
        # Finally, plot
        # plt.plot(x, y, c=col, ls=ls, label = f"{label} ($p = {p:.2f}$) ", linewidth = linewidth)
        this_label = f"{label} (${p:.2f}$)"
        saved_labels.append(this_label)
        saved_colors.append(col)
        saved_linestyles.append(ls)
        
        plt.plot(x, y, c=col, ls=ls, label = this_label, linewidth = linewidth)
        pvalues.append(p)
    
    ### Legend
    # OLD LEGEND
    # leg = plt.legend(loc = "upper left",
    #            ncol = 2,
    #            fontsize = 32, 
    #            handlelength = 1,
    #            columnspacing=1.0) # bbox_to_anchor = (1.025, 1.0), 
    # # change the line width for the legend
    # for line in leg.get_lines():
    #     line.set_linewidth(5.0)
    
    # Create custom legends for upper left and lower right
    # Splitting labels, colors, and linestyles into two parts
    upper_left_labels = saved_labels[:6]
    upper_left_colors = saved_colors[:6]
    upper_left_linestyles = saved_linestyles[:6]

    lower_right_labels = saved_labels[6:]
    lower_right_colors = saved_colors[6:]
    lower_right_linestyles = saved_linestyles[6:]

    # Create the handles
    lw_legend = 3.0
    lower_right_handles = [Line2D([0], [0], color=color, linestyle=linestyle, linewidth=lw_legend) for color, linestyle in zip(lower_right_colors, lower_right_linestyles)]
    
    upper_left_handles = [Line2D([0], [0], color=color, linestyle=linestyle, linewidth=lw_legend) for color, linestyle in zip(upper_left_colors, upper_left_linestyles)]

    legend_kwargs = {"fontsize": 32, 
                     "handlelength": 0.8,
                     "frameon": False,
                     "handletextpad": 0.5}

    eps_x = 0.0175
    eps_y = 0.02
    leg1 = plt.legend(upper_left_handles, upper_left_labels, loc = "upper left", bbox_to_anchor = (-eps_x, 1.0 + eps_y), **legend_kwargs)
    leg2 = plt.legend(lower_right_handles, lower_right_labels, loc = "lower right", bbox_to_anchor = (1.0 + eps_x, -eps_y), **legend_kwargs)
    
    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)


    plt.xlabel(r'Credible level')
    plt.ylabel(r'Fraction with credible level $\leq x$')
    print("Creating pp-plot, getting p values . . . DONE")

    print("pvalues")
    print(pvalues)
    ptotal = kstest(pvalues, cdf=uniform(0,1).cdf).pvalue
    string_total = f"N = {len(credible_level_list)}, Total p-value: {ptotal:.2f}"
    print(string_total)
    
    title_string = ""
    print(title_string)
    
    print("Saving pp-plot")
    plt.grid(False) # disable grida
    plt.title(title_string)
    xlim_eps = 1e-4
    plt.xlim(xlim_eps, 1)
    plt.ylim(xlim_eps, 1)
    
    for ext in [".png", ".pdf"]:
        full_savename = save_name + ext
        print(f"Saving pp-plot to: {full_savename}")
        plt.savefig(full_savename, bbox_inches="tight")

############
### MAIN ###
############

def main():
    
    compute = False
    convert_chi_eff = True
    exclude_bad_injections = False
    
    wf_names = ["TF2", "NRTv2"]
    for wf_name in wf_names:
        print(f"Waveform: {wf_name}")
        
        if compute:
            # Re-compute the p values for the injections
            result = get_credible_levels(wf_name, 
                                         convert_chi_eff=convert_chi_eff, 
                                         exclude_bad_injections=exclude_bad_injections)
        else:
            # Load with pickle
            with open(f"./percentiles/credible_levels_dict_{wf_name}.pkl", "rb") as f:
                result = pickle.load(f)
                
        # Compute p values:
        for which, result_values in result.items():
            print("Which: ", which)
            p_values = compute_p_value_kstest(result_values)
            for key, value in p_values.items():
                print(f"{key}: The p-value is {value}")
                
        print("Choosing which percentile calculation to use . . .")
        result = choose_percentile_dict(result)
        
        print("Making pp plot . . .")
        make_pp_plot(result, save_name=f"../figures/pp_plot_{wf_name}")
    
    
if __name__ == "__main__":
    main()