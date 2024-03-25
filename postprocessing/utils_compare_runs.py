"""
Some utilities for fetching the correct parameters etc that I don't want to clutter in the compare_runs.py script.
"""

import numpy as np
import jax.numpy as jnp
import h5py
import json
from ripple import get_chi_eff, Mc_eta_to_ms, lambda_tildes_to_lambdas

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

jim_naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

gwosc_names = ['chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'lambda_1', 'lambda_2', 'luminosity_distance', 't0', 'phase', 'iota', 'psi', 'ra', 'dec']
bilby_names = ['chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'lambda_1', 'lambda_2', 'luminosity_distance', 'phase', 'iota', 'psi', 'ra', 'dec']
trigger_time_GW190425 = 1240215503.017147
LABELS = [r'$\mathcal{M}_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

### RANGES ###

def get_ranges_GW170817_NRTidalv2(convert_chi, 
                                  convert_lambdas):
    
    if not convert_chi and not convert_lambdas:
        return [(1.197275, 1.19779),
                (0.55, 1.0),
                (-0.05, 0.049),
                (-0.05, 0.05),
                (0.0, 1700.0),
                (0.0, 2500.0),
                (10.0, 50.0),
                (0.0, 2 * np.pi),
                (1.75, np.pi),
                (0.0, np.pi),
                (3.35, 3.49),
                (-0.5, -0.2)]
        
    if convert_chi and not convert_lambdas:
        return [(1.197275, 1.19779),
                (0.55, 1.0),
                (-0.015, 0.049),
                (0.0, 1700.0),
                (0.0, 2500.0),
                (10.0, 50.0),
                (0.0, 2 * np.pi),
                (1.75, np.pi),
                (0.0, np.pi),
                (3.35, 3.49),
                (-0.5, -0.2)]
        
def get_ranges(event, convert_chi, convert_lambdas):
    if event == "GW170817_NRTidalv2":
        return get_ranges_GW170817_NRTidalv2(convert_chi, convert_lambdas)
    # TODO: implement the others as well here.
    
def get_idx_list_GW170817_NRTidalv2(n_dim: int = 12):
    if n_dim == 11:
        # idx_list = [1] * n_dim
        idx_list = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0] # 1 is jim
        # idx_list = [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1] # 0 is bilby
        assert len(idx_list) == n_dim, "Length of idx_list does not match n_dim in get_idx_list!"
        
    return idx_list

def get_idx_list(event, convert_chi, convert_lambdas):
    n_dim = 12
    if convert_chi:
        n_dim -= 1
    
    if event == "GW170817_NRTidalv2":
        return get_idx_list_GW170817_NRTidalv2(n_dim)
    

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
    
    return np.array(samples, dtype=np.float64)

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

def preprocess_samples(samples: np.array,
                       convert_chi: bool = True,
                       convert_lambdas: bool = False):
    
    mc, q, chi1, chi2, lambda1, lambda2 = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4], samples[:, 5]
    eta = q / (1 + q)**2
    m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
    
    if convert_lambdas:
        lambda_tilde, delta_lambda_tilde = lambda_tildes_to_lambdas(jnp.array([lambda1, lambda2, m1, m2]))
        
        samples[:, 4] = lambda_tilde
        samples[:, 5] = delta_lambda_tilde
        
    if convert_chi: 
        chi_eff = get_chi_eff(jnp.array([m1, m2, chi1, chi2]))
        samples = np.delete(samples, [2, 3], axis=1)
        samples = np.insert(samples, 2, chi_eff, axis=1)
        
    print("np.shape(samples)")
    print(np.shape(samples))
    
    return samples