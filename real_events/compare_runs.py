"""
Script to load and compare jim runs against Bilby runs
"""

"""
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

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$' ,r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

labels_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$' ,r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

gwosc_names = ['chirp_mass', 'mass_ratio', 'chi_eff', 'lambda_tilde', 'delta_lambda', 'luminosity_distance', 'phase', 'iota', 'psi', 'ra', 'dec']

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
        gwosc_indices = [pnames.index(name) for name in gwosc_names]
        
        print("parameter names:") 
        for name in pnames:
            print(name)

        # Fetch the posterior samples for the parameters that we are interested in
        samples = []
        for ind in gwosc_indices:
            samples.append([samp[ind] for samp in posterior[()]])

        samples = np.asarray(samples).T
        
        print("samples shape:", np.shape(samples))
        
    return samples

def weight_function(x):
    return x**2

# TODO improve getting the index right
def reweigh_distance(chains, d_idx = 6):
    """
    Get weights based on distance to mimic cosmological distance prior.
    """
    d_samples = chains[:, d_idx]
    print(d_samples)
    weights = weight_function(d_samples)
    weights = weights / np.sum(weights)
    
    return weights


############
### MAIN ###
############

def main():

    outdir = "./outdir/"
    corner_kwargs = default_corner_kwargs
    corner_kwargs["color"] = "blue"
    use_weights = False
    
    print(f"Reading data from {outdir}. Script paramters: use_weights={use_weights}")

    print("Reading bilby data")
    gwosc_samples = get_chains_GWOSC()
    gwosc_samples = gwosc_samples[:, [0,1,3,4]]

    print("Reading jim data")
    
    filename = f"{outdir}results_production.npz"
    data = np.load(filename)
    chains = data['chains'].reshape(-1, 13)
    chains[:,8] = np.arccos(chains[:,8])
    chains[:,11] = np.arcsin(chains[:,11])
    chains = np.asarray(chains)

    print("Loading data complete")

    ### Plot
    
    if use_weights:
        ### TODO - remove? Reweight based on distance
        # weights = reweigh_distance(chains)
        weights = 1 / len(chains) * np.ones(len(chains))
        name = f"peter_reweighted.png"
    
    if not use_weights:
        name = f"peter.png"
        weights = None
        
    print(f"Saving plot of chains to {name}")
    
    fig = corner.corner(chains, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
    corner_kwargs["color"] = "red"
    corner.corner(gwosc_samples, labels = labels, fig=fig, hist_kwargs={'density': True}, **corner_kwargs)
    fig.savefig(outdir + name, bbox_inches='tight')  
    
    print("DONE")
        
if __name__ == "__main__":
    # main()
    gwosc_filename = "/home/thibeau.wouters/gw-datasets/GW190425/posterior_samples.h5"
    gwosc_samples = get_chains_GWOSC(gwosc_filename)