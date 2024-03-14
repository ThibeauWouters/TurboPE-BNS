import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.prior import Uniform, PowerLaw, Composite 
import jax.numpy as jnp
import jax
import time
import numpy as np
jax.config.update("jax_enable_x64", True)
import shutil
import numpy as np
import matplotlib.pyplot as plt
import optax 
import sys
sys.path.append("../")
import utils_plotting as utils
print(jax.devices())

################
### PREAMBLE ###
################

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
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']
naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

data_path = "/home/thibeau.wouters/gw-datasets/GW190425/" # on CIT

start_runtime = time.time()

############
### BODY ###
############

### Data definitions

total_time_start = time.time()
gps = 1240215503.017147
trigger_time = gps
fmin = 20
fmax = 2048
minimum_frequency = fmin
maximum_frequency = fmax
T = 128
duration = T
post_trigger_duration = 2
epoch = duration - post_trigger_duration
f_ref = fmin
tukey_alpha = 2 / (T / 2)
print(tukey_alpha)

### Getting detector data

# Load the data
L1.load_data(trigger_time=trigger_time,
             gps_start_pad=duration-2,
             gps_end_pad=2,
             f_min=fmin,
             f_max=fmax,
             tukey_alpha = tukey_alpha,
             load_psd = False)

V1.load_data(trigger_time=trigger_time,
             gps_start_pad=duration-2,
             gps_end_pad=2,
             f_min=fmin,
             f_max=fmax,
             tukey_alpha = tukey_alpha,
             load_psd = False)

# # Load the PSDs from given files
# data_location = "/home/thibeau.wouters/gw-datasets/GW190425/"
# L1.load_psd(L1.frequencies, data_location + "glitch_median_PSD_forLI_L1_srate8192.txt")
# V1.load_psd(V1.frequencies, data_location + "glitch_median_PSD_forLI_V1_srate8192.txt")

# data_location = "./data/"

# data_dict = {"L1":{"data": data_path + "L-L1_HOFT_C01_T1700406_v3-1240211456-4096.gwf",
#                    "psd": data_path + "glitch_median_PSD_forLI_L1_srate8192.txt",
#                    "channel": "DCS-CALIB_STRAIN_CLEAN_C01_T1700406_v3"},
#             "V1":{"data": data_path + "V-V1Online_T1700406_v3-1240214000-2000.gwf",
#                     "psd": data_path + "glitch_median_PSD_forLI_V1_srate8192.txt",
#                     "channel": "Hrec_hoft_16384Hz_T1700406_v3"}
# }

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

L1.psd = L1.load_psd(L1.frequencies, data_path + "glitch_median_PSD_forLI_L1_srate8192.txt")
V1.psd = V1.load_psd(V1.frequencies, data_path + "glitch_median_PSD_forLI_V1_srate8192.txt")

### Define priors

# Internal parameters
Mc_prior = Uniform(1.485, 1.490, naming=["M_c"])
q_prior = Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1z_prior = Uniform(-0.05, 0.05, naming=["s1_z"])
s2z_prior = Uniform(-0.05, 0.05, naming=["s2_z"])
lambda_1_prior = Uniform(0.0, 5000.0, naming=["lambda_1"])
lambda_2_prior = Uniform(0.0, 5000.0, naming=["lambda_2"])
dL_prior       = Uniform(1.0, 500.0, naming=["d_L"])
t_c_prior      = Uniform(-0.1, 0.1, naming=["t_c"])
phase_c_prior  = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
cos_iota_prior = Uniform(
    -1.0,
    1.0,
    naming=["cos_iota"],
    transforms={
        "cos_iota": (
            "iota",
            lambda params: jnp.arccos(
                jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)
psi_prior     = Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior      = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
sin_dec_prior = Uniform(
    -1.0,
    1.0,
    naming=["sin_dec"],
    transforms={
        "sin_dec": (
            "dec",
            lambda params: jnp.arcsin(
                jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)

prior_list = [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        lambda_1_prior,
        lambda_2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]

prior = Composite(prior_list)

# The following only works if every prior has xmin and xmax property, which is OK for Uniform and Powerlaw
bounds = jnp.array([[p.xmin, p.xmax] for p in prior.priors])

### Create likelihood object

n_bins = 200

# ref_params = {
#     'M_c': 1.48597014,
#     'eta': 0.21593324,
#     's1_z': 0.03201217,
#     's2_z': -0.04464742,
#     'lambda_1': 3770.53191574,
#     'lambda_2': 12.0157453,
#     'd_L': 35.42152782,
#     't_c': -0.01343332,
#     'phase_c': 0.91422449,
#     'iota': 2.93582712,
#     'psi': 2.00726107,
#     'ra': 5.19030855,
#     'dec': 0.51985741
# }


ref_params = {
    'M_c': 1.486722,
    'eta': 0.18946014,
    's1_z': 0.04419246,
    's2_z': 0.00038679,
    'lambda_1': 455.74266717,
    'lambda_2': 144.29782064,
    'd_L': 131.97211914,
    't_c': -0.01579126,
    'phase_c': 1.98962121,
    'iota': 1.11046195,
    'psi': 2.02977615,
    'ra': 1.26495061,
    'dec': -0.42639091
}

likelihood = HeterodynedTransientLikelihoodFD([L1, V1], prior=prior, bounds=bounds, waveform=RippleTaylorF2(), trigger_time=gps, duration=T, n_bins=n_bins, ref_params=ref_params)
print("Running with n_bins  = ", n_bins)

# Local sampler args

eps = 1e-2
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

# Build the learning rate scheduler

n_loop_training = 400
n_epochs = 100
total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-4
power = 4.0
schedule_fn = optax.polynomial_schedule(
    start_lr, end_lr, power, total_epochs-start, transition_begin=start)

scheduler_str = f"polynomial_schedule({start_lr}, {end_lr}, {power}, {total_epochs-start}, {start})"

# Create jim object
outdir_name = "./outdir/"
jim = Jim(
    likelihood,
    prior,
    n_loop_training=n_loop_training,
    n_loop_production=40,
    n_local_steps=5,
    n_global_steps=400,
    n_chains=1000,
    n_epochs=n_epochs,
    learning_rate=schedule_fn,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=10,
    output_thinning=30,    
    local_sampler_arg=local_sampler_arg,
    stopping_criterion_global_acc = 0.10,
    outdir_name=outdir_name
)


# ### These were the old (december) hyperparams
# jim = Jim(
#     likelihood,
#     prior,
#     n_loop_training=200,
#     n_loop_production=200,
#     n_local_steps=500,
#     n_global_steps=500,
#     n_chains=2000,
#     n_epochs=100,
#     learning_rate=0.001,
#     max_samples=50000,
#     momentum=0.9,
#     batch_size=50000,
#     use_global=True,
#     keep_quantile=0.0,
#     train_thinning=20,
#     output_thinning=50,    
#     local_sampler_arg=local_sampler_arg,
#     outdir_name=outdir_name
# )

### Heavy computation begins
jim.sample(jax.random.PRNGKey(82))
### Heavy computation ends

# === Show results, save output ===

# Print a summary to screen:
jim.print_summary()
outdir = outdir_name

# Save and plot the results of the run
#  - training phase

name = outdir + f'results_training.npz'
print(f"Saving samples to {name}")
state = jim.Sampler.get_sampler_state(training=True)
chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state[
    "log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, log_prob=log_prob, local_accs=local_accs,
            global_accs=global_accs, loss_vals=loss_vals)

utils.plot_accs(local_accs, "Local accs (training)",
                "local_accs_training", outdir)
utils.plot_accs(global_accs, "Global accs (training)",
                "global_accs_training", outdir)
utils.plot_loss_vals(loss_vals, "Loss", "loss_vals", outdir)
utils.plot_log_prob(log_prob, "Log probability (training)",
                    "log_prob_training", outdir)

#  - production phase
name = outdir + f'results_production.npz'
state = jim.Sampler.get_sampler_state(training=False)
chains, log_prob, local_accs, global_accs = state["chains"], state[
    "log_prob"], state["local_accs"], state["global_accs"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, chains=chains, log_prob=log_prob,
            local_accs=local_accs, global_accs=global_accs)

utils.plot_accs(local_accs, "Local accs (production)",
                "local_accs_production", outdir)
utils.plot_accs(global_accs, "Global accs (production)",
                "global_accs_production", outdir)
utils.plot_log_prob(log_prob, "Log probability (production)",
                    "log_prob_production", outdir)

# Plot the chains as corner plots
utils.plot_chains(chains, "chains_production", outdir, truths=None)

# Save the NF and show a plot of samples from the flow
print("Saving the NF")
jim.Sampler.save_flow(outdir + "nf_model")
name = outdir + 'results_NF.npz'
chains = jim.Sampler.sample_flow(5_000)
np.savez(name, chains=chains)

# Final steps

# Finally, copy over this script to the outdir for reproducibility
shutil.copy2(__file__, outdir + "copy_script.py")

print("Saving the jim hyperparameters")
# Change scheduler from function to a string representation
try:
    jim.hyperparameters["learning_rate"] = scheduler_str
    jim.Sampler.hyperparameters["learning_rate"] = scheduler_str
    jim.save_hyperparameters(outdir=outdir)
except Exception as e:
    # Sometimes, something breaks, so avoid crashing the whole thing
    print(f"Could not save hyperparameters in script: {e}")

print("Finished successfully")

end_runtime = time.time()
runtime = end_runtime - start_runtime
print(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")

print(f"Saving runtime")
with open(outdir + 'runtime.txt', 'w') as file:
    file.write(str(runtime))