"""
Script to load and compare jim runs against Bilby runs
Here we compare our posterior samples with those obtained from GWOSC.

More information:
- GWOSC page for GW190425 results: https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190425/v3/
- Public posterior samples can be found here: https://dcc.ligo.org/public/0165/P2000026/002/posterior_samples.h5
"""

import time
import numpy as np
import matplotlib.pyplot as plt 
import corner
import h5py
import jax.numpy as jnp
import json
import copy
from scipy.spatial.distance import jensenshannon
import pickle

from ripple import get_chi_eff, Mc_eta_to_ms, lambda_tildes_to_lambdas
jim_naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

################
### PREAMBLE ###
################

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=24),
                        title_kwargs=dict(fontsize=24), 
                        color="blue",
                        # quantiles=[],
                        levels=[0.68, 0.95], # 0.997
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False
)

params = {
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "text.usetex": True,
    "font.family": "serif",
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
}
plt.rcParams.update(params)

# TODO have to change these
my_colors = {"jim": "blue", 
             "bilby": "gray"}

LABELS = [r'$\mathcal{M}_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

## TODO remove?
labels_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$' ,r'$d_{\rm{L}}/{\rm Mpc}$',r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

gwosc_names = ['chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'lambda_1', 'lambda_2', 'luminosity_distance', 't0', 'phase', 'iota', 'psi', 'ra', 'dec']
bilby_names = ['chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'lambda_1', 'lambda_2', 'luminosity_distance', 'phase', 'iota', 'psi', 'ra', 'dec']
trigger_time_GW190425 = 1240215503.017147 # for GW190425

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

def get_chains_GWOSC(filename: str, 
                     which_waveform: str = "TaylorF2",
                     remove_tc: bool = True) -> np.ndarray:
    """
    Retrieve posterior samples from the LIGO page: public posterior samples can be found here: https://dcc.ligo.org/public/0165/P2000026/002/posterior_samples.h5
    """
    
    allowed_waveforms = ["PhenomPNRT", "PhenomDNRT", "TaylorF2"]
    
    if which_waveform not in allowed_waveforms:
        space = "_"
        raise ValueError("Given waveform not recognized: should be in ", space.join(allowed_waveforms))
    
    which_waveform += "-LS" # limit to the low spin prior
    print("GWOSC: Fetching the data for waveform:", which_waveform)
    
    # Load the posterior samples from the HDF5 file
    with h5py.File(filename, 'r') as file:
        # Fetch indices of the names of parameters that we are interested in
        posterior = file[which_waveform]['posterior_samples']#[()]
        pnames = posterior.dtype.names
        # print("GWOSC parameter names:") 
        # for name in pnames:
        #     print(name)
        gwosc_indices = [pnames.index(name) for name in gwosc_names]

        # Fetch the posterior samples for the parameters that we are interested in
        samples = []
        for ind in gwosc_indices:
            samples.append([samp[ind] for samp in posterior[()]])

        samples = np.asarray(samples).T
        
        # Subtract trigger time from t0 samples
        t0_idx = gwosc_names.index("t0")
        samples[:, t0_idx] -= trigger_time_GW190425
        
        # Remove t_c if wanted:
        if remove_tc:
            samples = np.delete(samples, t0_idx, axis=1)
        
    return samples

def get_chains_bilby(filename: str,
                     convert_iota: bool = False) -> np.ndarray:
    """
    Retrieve posterior samples of an event from one of Peter's runs.
    
    Args:
        - filename: str, path to the results.json file
        
    Returns:
        - samples: np.ndarray, shape (n_samples, n_parameters)
    """
    
    with open(filename, 'r') as file:
        data = json.load(file)
        posterior_dict = data['posterior']['content']
        
        samples = [posterior_dict[key] for key in bilby_names]
        if convert_iota:
            cos_iota_index = bilby_names.index('cos_theta_jn')
            samples[cos_iota_index] = np.arccos(samples[cos_iota_index])
            
        # Iterate over the entries and convert dicts to arrays
        samples = np.array(samples)
        
        count = 0
        for i in range(len(samples)):
            for j in range(len(samples[i])):
                if isinstance(samples[i][j], dict):
                    count += 1
                    samples[i][j] = samples[i][j]["content"]
                
        samples = samples.T
        
    print("count")
    print(count)
    
    return samples

def get_chains_jim(filename: str,
                   remove_tc: bool = True) -> np.ndarray:
    data = np.load(filename)
    chains = data['chains'].reshape(-1, 13)
    cos_iota_index = jim_naming.index('cos_iota')
    sin_dec_index = jim_naming.index('sin_dec')
    chains[:, cos_iota_index] = np.arccos(chains[:, cos_iota_index])
    chains[:, sin_dec_index] = np.arcsin(chains[:, sin_dec_index])
    
    if remove_tc:
        tc_index = jim_naming.index('t_c')
        chains = np.delete(chains, tc_index, axis=1)
        
    chains = np.asarray(chains)
    
    print("np.shape(chains) jim")
    print(np.shape(chains))
    
    return chains

def get_idx_list(n_dim):
    # TODO: make this automatic
    idx_list = [1 for _ in range(n_dim)]
    return idx_list

def get_plot_samples(posterior_list: list[np.array],
                     idx_list: list[int] = []):
    
    plotsamples_list = []
    n_dim = len(posterior_list[0].T)
    
    sizes = [len(posterior) for posterior in posterior_list]
    smallest_size = min(sizes)
    
    # Resample posteriors for the same size
    for i, posterior in enumerate(posterior_list):
        this_size = sizes[i]
        all_idx = np.arange(this_size)
        sampled_idx = np.random.choice(all_idx, size=smallest_size, replace=False)
        sampled_posterior = posterior[sampled_idx]
        plotsamples_list.append(sampled_posterior)
        # sampled_xs_list.append(sampled_posterior[params].values)
        
    idx_list = get_idx_list(n_dim)
        
    # Create plotsamples dummy
    print("np.shape(plotsamples_list)")
    print(np.shape(plotsamples_list))
    
    plotsamples_dummy = np.vstack([plotsamples_list[idx_list[i]] for i in range(len(idx_list))])
    
    print("np.shape(plotsamples_list)")
    print(np.shape(plotsamples_list))
    print("np.shape(plotsamples_dummy)")
    print(np.shape(plotsamples_dummy))
    
    return plotsamples_list, plotsamples_dummy


def plot_comparison(jim_path, 
                    bilby_path, 
                    use_weights = False,
                    save_name = "corner_comparison",
                    which_waveform: str = "TaylorF2",
                    remove_tc: bool = True,
                    **corner_kwargs):
    
    start_time = time.time()
    
    print("Reading bilby data")
    if ".h5" in bilby_path:
        bilby_samples = get_chains_GWOSC(bilby_path, which_waveform=which_waveform)
    else:
        bilby_samples = get_chains_bilby(bilby_path)

    print("Reading jim data")
    jim_samples = get_chains_jim(jim_path, remove_tc = remove_tc)

    print("Loading data complete")

    # If wanted, reweigh uniform samples to powerlaw samples
    if use_weights:
        weights = reweigh_distance(jim_samples)
        save_name += "_reweighted"
    
    if not use_weights:
        weights = None
    
    # TODO: to chi eff here? 
    
    # Remove the t_c label for comparison with bilby
    if remove_tc:
        labels = copy.deepcopy(LABELS)
        labels.remove(r'$t_c$')
    else:
        labels = LABELS
        
    print(f"Saving plot of chains to {save_name}")
    
    plotsamples_list, dummy_values = get_plot_samples([jim_samples, bilby_samples])
    
    # Dummy postprocessing
    corner_kwargs["color"] = "white"
    fig = corner.corner(dummy_values, alpha=0, plot_contours=False, hist_kwargs={'density': True, 'alpha': 0}, **corner_kwargs)
    
    # Actual plotting
    corner_kwargs["color"] = my_colors["jim"]
    for samples, color in zip(plotsamples_list, [my_colors["jim"], my_colors["bilby"]]):
        corner_kwargs["color"] = color
        corner.corner(samples, labels = labels, fig=fig, weights=weights, hist_kwargs={'density': True}, **corner_kwargs)
    
    # # Dummy postprocessing
    # corner_kwargs["color"] = "white"
    # corner.corner(dummy_values, fig=fig, plot_contours=False, alpha=0, hist_kwargs={'density': True, 'alpha': 0}, **corner_kwargs)
    
    # TODO improve the plot, e.g. give a custom legend
    for ext in ["png", "pdf"]:
        plt.savefig(f"{save_name}.{ext}", bbox_inches='tight')
    plt.close()
    
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time}")

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
    peter_samples = get_chains_bilby(peter_path)
    
    # Load GWOSC samples
    gwosc_samples = get_chains_GWOSC(gwosc_path, which_waveform=which_waveform)
    
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
        
        js_div = jensenshannon(histogram_jim, histogram_bilby) ** 2
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

def main():
    
    gwosc_path = "/home/thibeau.wouters/gw-datasets/GW190425/posterior_samples.h5"
    paths_dict = {"GW170817_TaylorF2": {"jim": "/home/thibeau.wouters/TurboPE-BNS/real_events/GW170817_TaylorF2/outdir_copy/results_production.npz",
                      "bilby": "/home/thibeau.wouters/jim_pbilby_samples/GW170817/GW170817-TF2_rejection_sampling_result.json"},
                  
                  "GW170817_NRTidalv2": {"jim": "/home/thibeau.wouters/TurboPE-BNS/real_events/GW170817_NRTidalv2/outdir/results_production.npz",
                      "bilby": "/home/thibeau.wouters/jim_pbilby_samples/GW170817/GW170817_IMRDNRTv2_older_bilby_result.json",
                      },
                  
                  "GW190425_TaylorF2": {"jim": "/home/thibeau.wouters/TurboPE-BNS/real_events/GW190425_TaylorF2_redo/outdir/results_production.npz",
                      "bilby": "/home/thibeau.wouters/jim_pbilby_samples/GW190425/GW190425-TF2_result.json",
                      },
                  
                  "GW190425_NRTidalv2": {"jim": "/home/thibeau.wouters/TurboPE-BNS/real_events/GW190425_NRTidalv2/outdir/results_production.npz",
                      "bilby": "/home/thibeau.wouters/jim_pbilby_samples/GW190425/GW190425_IMRDNRTv2_older_bilby_result.json",
                      },
                  
                  "GW190425_TaylorF2_online_data": {"jim": "/home/thibeau.wouters/TurboPE-BNS/real_events/GW190425_TaylorF2_redo/outdir/results_production.npz",
                               "bilby": gwosc_path,
                               },
                  
                  "GW190425_NRTidalv2_online_data": {"jim": "/home/thibeau.wouters/TurboPE-BNS/real_events/GW190425_NRTidalv2/outdir/results_production.npz",
                               "bilby": gwosc_path,
                               }}
    
    ranges_dict = {"GW170817_NRTidalv2": [(1.197275, 1.1978),
                                         (0.55, 1.0),
                                         (-0.05, 0.05),
                                         (-0.05, 0.05),
                                         (0.0, 1700.0),
                                         (0.0, 2500.0),
                                         (10.0, 50.0),
                                         (0.0, 2 * np.pi),
                                         (1.75, np.pi),
                                         (0.0, np.pi),
                                         (3.35, 3.50),
                                         (-0.5, -0.2)]
    }
                                         
    save_path = "../figures/"
    
    # events_to_plot = ["GW170817_TaylorF2",
    #                   "GW170817_NRTidalv2",
    #                   "GW190425_TaylorF2",
    #                   "GW190425_NRTidalv2"]
    
    events_to_plot = ["GW170817_NRTidalv2"]
    
    for event, paths in paths_dict.items():
        
        # Skip the paths that we are not going to plot
        if event not in events_to_plot:
            continue
        
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
        corner_kwargs["range"] = ranges_dict[event]
        
        plot_comparison(jim_path, 
                        bilby_path, 
                        use_weights = False,
                        save_name = save_path + event,
                        which_waveform = which_waveform,
                        remove_tc = True,
                        **corner_kwargs)
        
        # ====== Computing the JS divergences ======
        
        # jim_chains = get_chains_jim(jim_path)
        # bilby_chains = get_chains_bilby(bilby_path)
        
        # js_dict = compute_js_divergences(jim_chains,
        #                                  bilby_chains, 
        #                                  plot_name = event)
        
        # print(js_dict)
        
        
    # ====== Compare the bilby runs ======
        
    # paths_dict_bilby_comparison = {"GW190425_TaylorF2": {"peter": "/home/thibeau.wouters/jim_pbilby_samples/GW190425/GW190425-TF2_result.json",
    #                                                      "gwosc": gwosc_path},
                                   
    #                                "GW190425_NRTidalv2": {"peter": "/home/thibeau.wouters/jim_pbilby_samples/GW190425/GW190425-IMRDNRTv2_result.json",
    #                                                      "gwosc": gwosc_path}
                  
    #               }
    
    # print("Comparing the bilby runs")
    # for event in paths_dict_bilby_comparison:
    #     print("==============================================")
    #     print(f"Comparing runs for: {event}")
    #     print("==============================================")
    #     peter_path = paths_dict_bilby_comparison[event]["peter"]
    #     gwosc_path = paths_dict_bilby_comparison[event]["gwosc"]
        
    #     if "TaylorF2" in peter_path:
    #         which_waveform = "TaylorF2"
    #     else:
    #         which_waveform = "PhenomDNRT"
        
    #     print(f"which waveform is set to {which_waveform}")
        
    #     compare_bilby_runs(peter_path, 
    #                        gwosc_path, 
    #                        save_name = save_path + event,
    #                        which_waveform = which_waveform,
    #                        remove_tc = True)
    
    # print("DONE")
        
if __name__ == "__main__":
    main()