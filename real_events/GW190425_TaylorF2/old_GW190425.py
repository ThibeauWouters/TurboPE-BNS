import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# jim
from jimgw.jim import Jim
from jimgw.detector import L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.waveform import RippleTaylorF2
from jimgw.prior import Uniform
from jimgw.fisher_information_matrix import FisherInformationMatrix
# ripple
# flowmc
from flowMC.utils.PRNG_keys import initialize_rng_keys
# jax
import jax.numpy as jnp
import jax

# others
import time
import numpy as np
from lal import GreenwichMeanSiderealTime
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

import shutil
import numpy as np
import matplotlib.pyplot as plt
import corner

# TODO move!!!
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

params = {
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(params)

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

### Data definitions

total_time_start = time.time()
trigger_time = 1240215503.017147
gps = trigger_time
fmin = 20
fmax = 2048
minimum_frequency = fmin
maximum_frequency = fmax
T = 128
duration = T
post_trigger_duration = 2
epoch = duration - post_trigger_duration
f_ref = fmin 
# gmst = GreenwichMeanSiderealTime(trigger_time)
# gsmt = Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad

### Getting detector data

data_location = "./data/"

data_dict = {"L1":{"data": data_location + "L-L1_HOFT_C01_T1700406_v3-1240211456-4096.gwf",
                   "psd": data_location + "glitch_median_PSD_forLI_L1_srate8192.txt",
                   "channel": "DCS-CALIB_STRAIN_CLEAN_C01_T1700406_v3"},
            "V1":{"data": data_location + "V-V1Online_T1700406_v3-1240214000-2000.gwf",
                    "psd": data_location + "glitch_median_PSD_forLI_V1_srate8192.txt",
                    "channel": "Hrec_hoft_16384Hz_T1700406_v3"}
}

# L1.load_data_from_frame(trigger_time=trigger_time,
#                         gps_start_pad=duration-2,
#                         gps_end_pad=2,
#                         frame_file_path=data_dict["L1"]["data"],
#                         channel_name=data_dict["L1"]["channel"],
#                         f_min=fmin,
#                         f_max=fmax)

# V1.load_data_from_frame(trigger_time=trigger_time,
#                         gps_start_pad=duration-2,
#                         gps_end_pad=2,
#                         frame_file_path=data_dict["V1"]["data"],
#                         channel_name=data_dict["V1"]["channel"],
#                         f_min=fmin,
#                         f_max=fmax)

L1.load_data(trigger_time=trigger_time,
             gps_start_pad=duration-2,
             gps_end_pad=2,
             f_min=fmin,
             f_max=fmax,
             tukey_alpha = 0.015625,
             load_psd=False)

V1.load_data(trigger_time=trigger_time,
             gps_start_pad=duration-2,
             gps_end_pad=2,
             f_min=fmin,
             f_max=fmax,
             tukey_alpha = 0.015625,
             load_psd=False)

L1.load_psd_from_file(data_dict["L1"]["psd"])
V1.load_psd_from_file(data_dict["V1"]["psd"])
    
# Prior
prior = Uniform(
    xmin=[1.485, 0.34, -0.05, -0.05,    0.0, -500.0,   1.0, -0.1,        0.0, -1.0,    0.0,        0.0, -1],
    xmax=[1.490,  1.0,  0.05,  0.05, 1200.0,  500.0, 500.0,  0.1, 2 * jnp.pi,  1.0, jnp.pi, 2 * jnp.pi,  1],
    naming=[
        "M_c",
        "q",
        "s1_z", 
        "s2_z", 
        "lambda_tilde",
        "delta_lambda_tilde",
        "d_L",
        "t_c",
        "phase_c",
        "cos_iota",
        "psi",
        "ra",
        "sin_dec",
    ],
    transforms = {"q": ("eta", lambda params: params['q']/(1+params['q'])**2),
                 "cos_iota": ("iota",lambda params: jnp.arccos(jnp.arcsin(jnp.sin(params['cos_iota']/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi))}
)

### Create likelihood object


ref_params = {
    'M_c': 1.486722,
    'eta': 0.18946014,
    's1_z': 0.04419246,
    's2_z': 0.00038679,
    'lambda_tilde': 455.74266717,
    'delta_lambda_tilde': -144.29782064,
    'd_L': 131.97211914,
    't_c': -0.01579126,
    'phase_c': 1.98962121,
    'iota': 1.11046195,
    'psi': 2.02977615,
    'ra': 1.26495061,
    'dec': -0.42639091
}


likelihood = HeterodynedTransientLikelihoodFD([L1, V1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=RippleTaylorF2(), trigger_time=gps, duration=T, n_bins=500, ref_params=ref_params)

### Create sampler and jim objects

# Mass matrix (this is copy pasted from the TurboPE set up)
# TODO get automated mass matrix, or make sure this doesn't break things
eps = 1e-3
n_chains = 1000
n_dim = 13
mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[7,7].set(1e-5)
mass_matrix = mass_matrix.at[11,11].set(1e-2)
mass_matrix = mass_matrix.at[12,12].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * eps}

# ### Check the Fisher information matrix for autotuning the mass matrix
# fim = FisherInformationMatrix([H1, L1, V1], waveform=RippleTaylorF2(), trigger_time=gps, duration=T, post_trigger_duration=post_trigger_duration)
# tuned_mass_matrix = fim.tune_mass_matrix(prior, RippleTaylorF2(), ref_params, H1.frequencies)

# print("tuned_mass_matrix")
# print(jnp.diag(tuned_mass_matrix))

outdir_name = "./outdir/"

jim = Jim(
    likelihood,
    prior,
    n_loop_pretraining=0,
    n_loop_training=200,
    n_loop_production=200,
    n_local_steps=500,
    n_global_steps=500,
    n_chains=2000,
    n_epochs=100,
    learning_rate=0.001,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=20,
    output_thinning=50,    
    local_sampler_arg=local_sampler_arg,
    outdir_name=outdir_name
)

### Heavy computation begins
jim.sample(jax.random.PRNGKey(37))
### Heavy computation ends

# === Show results, save output ===

# Cleaning outdir
for filename in os.listdir(outdir_name):
    file_path = os.path.join(outdir_name, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

### Summary
jim.print_summary()

### Diagnosis plots of summaries
print("Creating plots")
jim.Sampler.plot_summary("training")
jim.Sampler.plot_summary("production")

# TODO - save the NF object to sample from later on
# print("Saving jim object (Normalizing flow)")
# jim.Sampler.save_flow("my_nf_IMRPhenomD")

### Write to 
which_list = ["training", "production"]
for which in which_list:
    name = outdir_name + f'results_{which}.npz'
    print(f"Saving {which} samples in npz format to {name}")
    state = jim.Sampler.get_sampler_state(which)
    chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
    np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

print("Sampling from the flow")
chains = jim.Sampler.sample_flow(10000)
name = outdir_name + 'results_NF.npz'
print(f"Saving flow samples to {name}")
np.savez(name, chains=chains)

### Plot chains and samples

# Production samples:
file = outdir_name + "results_production.npz"
name = outdir_name + "results_production.png"

data = np.load(file)
# TODO improve the following: ignore t_c, and reshape with n_dims, and do conversions
idx_list = [0,1,2,3,4,5,6,8,9,10,11,12]
chains = data['chains'][:,:,idx_list].reshape(-1,12)
chains[:,8] = np.arccos(chains[:,8])
chains[:,11] = np.arcsin(chains[:,11])
chains = np.asarray(chains)
corner_kwargs = default_corner_kwargs
fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig(name, bbox_inches='tight')  

# Production samples:
file = outdir_name + "results_NF.npz"
name = outdir_name + "results_NF.png"

data = np.load(file)["chains"]
print("np.shape(data)")
print(np.shape(data))

# TODO improve the following: ignore t_c, and reshape with n_dims, and do conversions
chains = data[:, idx_list]
# chains[:,6] = np.arccos(chains[:,6])
# chains[:,9] = np.arcsin(chains[:,9]) # TODO not sure if this is still necessary?
chains = np.asarray(chains)
corner_kwargs = default_corner_kwargs
fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig(name, bbox_inches='tight')  
    
    
print("Saving the hyperparameters")
jim.save_hyperparameters()