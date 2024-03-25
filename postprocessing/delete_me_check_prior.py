import numpy as np
import matplotlib.pyplot as plt 
import h5py

filename = "/home/gregory.ashton/public_html/O3/pe_O3_S190425z_git_repo/pe_samples/PROD9_posterior_samples.hdf5" # on CIT

with h5py.File(filename, 'r') as file:
    posterior = file["lalinference"]["lalinference_nest"]["posterior_samples"]
    # Fetch indices of the names of parameters that we are interested in
    # posterior = file["PhenomDNRT-LS"]['posterior_samples']#[()]
    pnames = posterior.dtype.names
    print("GWOSC parameter names:") 
    for name in pnames:
        print(name)
    # # gwosc_indices = [pnames.index(name) for name in gwosc_names]