import os
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

outdir_dict = {"TF2": "../injections/outdir_TF2/",
               "NRTv2": "../injections/outdir_NRTv2/"}

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Iterate over the 2 waveforms
for key, outdir in outdir_dict.items():
    print(f"Checking dL distribution for {key}")
    # Iterate over the directories
    dL_values = []
    for subdir in os.listdir(outdir):
        if os.path.isdir(outdir + subdir):
            path = outdir + subdir + "/config.json"
            with open(path) as f:
                config = json.load(f)
            dL_values.append(config["d_L"])
            
    dL_values = np.array(dL_values)

    # Get samples from a uniform range to compare
    dL_min = 30.0
    dL_max = 300.0

    # Get KDE
    x = np.linspace(dL_min, dL_max, 1_000)
    kde = gaussian_kde(dL_values)
    kde_uniform = lambda x: 1/(dL_max - dL_min) * np.ones_like(x)

    # Plot the histogram of the dL values
    lw = 2
    nb_bins = 10
    hist_injections, edges = np.histogram(dL_values, bins=nb_bins, density=True)
    hist_uniform = np.ones_like(hist_injections) / (dL_max - dL_min)
    # plt.hist(dL_values, bins=20, histtype='step', density=True, label="Samples", linewidth=lw)
    # plt.hist(dL_uniform, bins=20, histtype='step', density=True, label="Uniform", linewidth=lw)
    
    # Make the plot
    plt.figure(figsize=(12, 7))
    plt.stairs(hist_injections, edges, label="Samples", linewidth=lw)
    plt.stairs(hist_uniform, edges, label="Uniform", linewidth=lw)
    plt.plot(x, kde(x), label="KDE", linewidth=lw)
    plt.xlabel("dL")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"../figures/dL_histogram_{key}.png")
    plt.close()

    # Save the KDE object
    kde_file = f"./kde_dL_{key}.npz"
    np.savez(kde_file, x=x, y=kde(x))