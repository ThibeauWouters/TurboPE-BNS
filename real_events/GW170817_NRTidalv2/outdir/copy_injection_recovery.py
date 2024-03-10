import psutil
p = psutil.Process()
p.cpu_affinity([0])

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

# jim
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.prior import Uniform, PowerLaw, Composite
# ripple
# flowmc
from flowMC.utils.PRNG_keys import initialize_rng_keys
# jax
import jax.numpy as jnp
import jax
# others
import numpy as np
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import corner

import optax
import utils

print(jax.devices())

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


load_online_data = False
labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

### Data definitions

total_time_start = time.time()
gps = 1187008882.43
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
waveform = RippleIMRPhenomD_NRTidalv2(f_ref=f_ref)

### Getting detector data

H1_frequency, H1_data_re, H1_data_im = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_fd_strain.txt').T
H1_data = H1_data_re + 1j*H1_data_im
H1_psd_frequency, H1_psd = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_psd.txt').T

H1_data = H1_data[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_psd = H1_psd[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_frequency = H1_frequency[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]

L1_frequency, L1_data_re, L1_data_im = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_fd_strain.txt').T
L1_data = L1_data_re + 1j*L1_data_im
L1_psd_frequency, L1_psd = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_psd.txt').T

L1_data = L1_data[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_psd = L1_psd[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_frequency = L1_frequency[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]

V1_frequency, V1_data_re, V1_data_im = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_fd_strain.txt').T
V1_data = V1_data_re + 1j*V1_data_im
V1_psd_frequency, V1_psd = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_psd.txt').T

V1_data = V1_data[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]
V1_psd = V1_psd[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]
V1_frequency = V1_frequency[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]

### Getting ifos and overwriting with above data

if load_online_data:
    tukey_alpha = 2 / (duration / 2)

    H1.load_data(gps, duration, 2, fmin, fmax, psd_pad=16, tukey_alpha=tukey_alpha)
    L1.load_data(gps, duration, 2, fmin, fmax, psd_pad=16, tukey_alpha=tukey_alpha)
    V1.load_data(gps, duration, 2, fmin, fmax, psd_pad=16, tukey_alpha=tukey_alpha)

    H1.load_psd_from_file('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_psd.txt')
    L1.load_psd_from_file('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_psd.txt')
    V1.load_psd_from_file('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_psd.txt')

else:
    H1.frequencies = H1_frequency
    H1.data = H1_data
    H1.psd = H1_psd 

    L1.frequencies = L1_frequency
    L1.data = L1_data
    L1.psd = L1_psd 

    V1.frequencies = V1_frequency
    V1.data = V1_data
    V1.psd = V1_psd 
    
assert np.allclose(H1_frequency, L1_frequency), "Frequencies are not the same for H1 and L1"
assert np.allclose(H1_frequency, V1_frequency), "Frequencies are not the same for H1 and V1"

frequencies = H1_frequency

### Define priors

# Internal parameters
Mc_prior = Uniform(1.18, 1.21, naming=["M_c"])
q_prior = Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1z_prior     = Uniform(-0.05, 0.05, naming=["s1_z"])
s2z_prior     = Uniform(-0.05, 0.05, naming=["s2_z"])
lambda1_prior = Uniform(0.0, 5000.0, naming=["lambda_1"])
lambda2_prior = Uniform(0.0, 5000.0, naming=["lambda_2"])

# External parameters
# dL_prior       = PowerLaw(1.0, 75.0, 2.0, naming=["d_L"])
dL_prior       = Uniform(1.0, 75.0, naming=["d_L"])
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

prior = Composite([
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        lambda1_prior,
        lambda2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]
)

# The following only works if every prior has xmin and xmax property, which is OK for Uniform and Powerlaw
bounds = jnp.array([[p.xmin, p.xmax] for p in prior.priors])

### Create likelihood object
ref_params = {'M_c': 1.19754357, 
              'eta': 0.24984541, 
              's1_z': -0.00429651, 
              's2_z': 0.00470304, 
              'lambda_1': 1816.51300368, 
              'lambda_2': 0.10161503, 
              'd_L': 10.87770389, 
              't_c': 0.00864911, 
              'phase_c': 4.33436689, 
              'iota': 1.59216065, 
              'psi': 1.69112445, 
              'ra': 5.08658471, 
              'dec': 0.47136332
}
ref_params = None
n_bins = 100
likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=bounds, waveform=waveform, trigger_time=gps, duration=T, n_bins=n_bins, ref_params=ref_params)

print("Running with n_bins  = ", n_bins)

### Create sampler and jim objects

eps = 1e-5
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

outdir_name = "./outdir/"
outdir = outdir_name

n_epochs = 50
n_loop_training = 100

total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-5
power = 4.0
schedule_fn = optax.polynomial_schedule(
    start_lr, end_lr, power, total_epochs-start, transition_begin=start)
scheduler_string = f"Polynomial scheduler: start_lr = {start_lr}, end_lr = {end_lr}, power = {power}, start = {start}"


jim = Jim(
    likelihood,
    prior,
    n_loop_training=400,
    n_loop_production=20,
    n_local_steps=5,
    n_global_steps=400,
    n_chains=n_chains,
    n_epochs=50,
    learning_rate=schedule_fn,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=10,
    output_thinning=30,    
    n_loops_maximize_likelihood = 2000,
    local_sampler_arg=local_sampler_arg,
    stopping_criterion_global_acc = 0.2,
    outdir_name=outdir_name
)

start = time.time()
### Heavy computation begins
jim.sample(jax.random.PRNGKey(42))
### Heavy computation ends
end = time.time()

# Print time in minutes
print("Total time taken = ", (end - start) / 60, " minutes")

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

# Overwrite the learning rate
try:
    jim.Sampler.hyperparameters["learning_rate"] = schedule_fn
except Exception as e:
    print(f"Failed to overwrite learning rate: {e}")

# === Show results, save output ===

# # Cleaning outdir
# for filename in os.listdir(outdir):
#     file_path = os.path.join(outdir_name, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))

### Summary
jim.print_summary()

### Diagnosis plots of summaries

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
chains = jim.Sampler.sample_flow(10_000)
np.savez(name, chains=chains)
# TODO debug this
# utils.plot_chains(chains, "chains_NF", outdir, truths=truths)

# Final steps

# Finally, copy over this script to the outdir for reproducibility
shutil.copy2(__file__, outdir + "copy_injection_recovery.py")

print("Saving the jim hyperparameters")
jim.save_hyperparameters(outdir=outdir)

end_time = time.time()
runtime = end_time - total_time_start
print(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")

print(f"Saving runtime")
with open(outdir + 'runtime.txt', 'w') as file:
    file.write(str(runtime))

print("Finished successfully!!!")