"""
Script to load and compare jim runs against Bilby runs
Here we compare our posterior samples with those obtained from GWOSC.

More information:
- GWOSC page for GW190425 results: https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190425/v3/
- Posterior samples can be found here: https://dcc.ligo.org/public/0165/P2000026/002/posterior_samples.h5
"""

import numpy as np
import matplotlib.pyplot as plt 
import corner
import h5py
import jax.numpy as jnp

from ripple import get_chi_eff, Mc_eta_to_ms, lambda_tildes_to_lambdas
naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

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
                        # levels=[0.9],
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
my_colors = ["blue", "orange"]

labels = [r'$\mathcal{M}_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$' ,r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

## TODO remove?
labels_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$' ,r'$d_{\rm{L}}/{\rm Mpc}$',r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

gwosc_names = ['chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'lambda_tilde', 'delta_lambda', 'luminosity_distance', 't0', 'phase', 'iota', 'psi', 'ra', 'dec']

#################
### UTILITIES ###
#################

def get_chains_GWOSC(filename: str, key: str = "TaylorF2"):
    """
    Retrieve posterior samples of an event
    """
    
    if key not in ["PhenomPNRT", "TaylorF2"]:
        raise ValueError("key should be PhenomPNRT or TaylorF2")
    
    key += "-LS" # we only consider low spin
    
    # Load the posterior samples from the HDF5 file
    with h5py.File(filename, 'r') as file:
        print('Top-level data structures:',file.keys())

        # Fetch indices of the names of parameters that we are interested in
        posterior = file[key]['posterior_samples']#[()]
        pnames = posterior.dtype.names
        print("GWOSC parameter names:") 
        for name in pnames:
            print(name)
        gwosc_indices = [pnames.index(name) for name in gwosc_names]

        # Fetch the posterior samples for the parameters that we are interested in
        samples = []
        for ind in gwosc_indices:
            samples.append([samp[ind] for samp in posterior[()]])

        samples = np.asarray(samples).T
        
        print("samples shape:", np.shape(samples))
        
    return samples

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

def plot_comparison(jim_path, 
                    gwosc_path, 
                    use_weights = False,
                    save_name = "corner_comparison"):
    
    corner_kwargs = default_corner_kwargs
    corner_kwargs["color"] = my_colors[0]
    
    print("Reading bilby data")
    gwosc_samples = get_chains_GWOSC(gwosc_path)

    print("Reading jim data")

    data = np.load(jim_path)
    chains = data['chains'].reshape(-1, 13)
    cos_iota_index = naming.index('cos_iota')
    sin_dec_index = naming.index('sin_dec')
    chains[:, cos_iota_index] = np.arccos(chains[:, cos_iota_index])
    chains[:, sin_dec_index] = np.arcsin(chains[:, sin_dec_index])
    chains = np.asarray(chains)

    print("Loading data complete")

    ### Plot
    
    if use_weights:
        weights = reweigh_distance(chains)
        # weights = 1 / len(chains) * np.ones(len(chains))
        save_name += "_reweighted"
    
    if not use_weights:
        weights = None
        
    print(f"Saving plot of chains to {save_name}")
    
    fig = corner.corner(chains, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
    corner_kwargs["color"] = my_colors[1]
    corner.corner(gwosc_samples, labels = labels, fig=fig, hist_kwargs={'density': True}, **corner_kwargs)
    
    # TODO improve the plot, e.g. give a custom legend
    for ext in ["png", "pdf"]:
        plt.savefig(f"{save_name}.{ext}", bbox_inches='tight')

############
### MAIN ###
############

def main():

    save_path = "../figures/"
    
    paths_dict = {"GW190425_online_data": {"jim": "/home/thibeau.wouters/TurboPE-BNS/real_events/GW190425_TaylorF2_online_data/outdir/results_production.npz",
                               "bilby": "/home/thibeau.wouters/gw-datasets/GW190425/posterior_samples.h5",
                               }}
    
    paths_dict = {"GW190425_TaylorF2": {"jim": "/home/thibeau.wouters/TurboPE-BNS/real_events/GW190425_TaylorF2/outdir/results_production.npz",
                               "bilby": "/home/thibeau.wouters/gw-datasets/GW190425/posterior_samples.h5",
                               }}

    for event, paths in paths_dict.items():
        print(f"Comparing runs for: {event}")
        jim_path = paths["jim"]
        bilby_path = paths["bilby"]
        plot_comparison(jim_path, 
                        bilby_path, 
                        use_weights = True,
                        save_name = save_path + event)
    
    print("DONE")
        
if __name__ == "__main__":
    main()