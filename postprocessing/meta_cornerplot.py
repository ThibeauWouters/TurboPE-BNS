import psutil
p = psutil.Process()
p.cpu_affinity([0])

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt 
import corner
import copy

import utils_compare_runs
from utils_compare_runs import paths_dict, LABELS

params = {
    # "axes.labelsize": 132,
    # "axes.titlesize": 132,
    "text.usetex": True,
    "font.family": "times new roman",
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
plt.rcParams.update(params)

label_fontsize = 26
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=label_fontsize),
                        title_kwargs=dict(fontsize=label_fontsize), 
                        # quantiles=[],
                        levels=[0.68, 0.95], # 0.997 # for 3 sigma as well?
                        plot_density=False,
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        labelpad = 0.25
)

COLORS = {"pbilby": "red", 
          "jim": "blue", 
          "RB": "green", 
          "ROQ": "brown"}

def main():
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    # Plot all on top of each other
    all_events = ["GW170817_TaylorF2",
                  "GW170817_NRTidalv2",
                  "GW190425_TaylorF2",
                  "GW190425_NRTidalv2"]
    
    plt.figure() 
    hist_kwargs = {"density": True}
    for event in all_events:
        print(f"Making the meta corner plot for {event} . . .")
        # Pbilby
        path = paths_dict[event]["bilby"]
        chains = utils_compare_runs.get_chains_bilby(path)
        
        corner_kwargs["color"] = COLORS["pbilby"]
        hist_kwargs["color"] = COLORS["pbilby"]
        corner_kwargs["hist_kwargs"] = hist_kwargs
        fig = corner.corner(chains, labels=LABELS,**corner_kwargs)
            
        # Jim:
        path = paths_dict[event]["jim"]
        chains = utils_compare_runs.get_chains_jim(path)
        corner_kwargs["color"] = COLORS["jim"]
        hist_kwargs["color"] = COLORS["jim"]
        corner_kwargs["hist_kwargs"] = hist_kwargs
        corner.corner(chains, fig=fig, **corner_kwargs)
        
        # RB:
        path = utils_compare_runs.bilby_RB_paths_dict[event]
        chains = utils_compare_runs.get_chains_hdf5(path)
        corner_kwargs["color"] = COLORS["RB"]
        hist_kwargs["color"] = COLORS["RB"]
        corner_kwargs["hist_kwargs"] = hist_kwargs
        corner.corner(chains, fig=fig, **corner_kwargs)
        
        # ROQ: 
        try: 
            path = utils_compare_runs.bilby_ROQ_paths_dict[event]
            chains = utils_compare_runs.get_chains_hdf5(path)
            corner_kwargs["color"] = COLORS["ROQ"]
            hist_kwargs["color"] = COLORS["ROQ"]
            corner_kwargs["hist_kwargs"] = hist_kwargs
            corner.corner(chains, fig=fig, **corner_kwargs)
        except Exception as e:
            print("This event is not supported by ROQ yet. Full error message follows now:")
            print(e)
        
        plt.savefig(f"../figures/meta_corner_{event}.png")
        plt.close()
    
if __name__ == "__main__":
    main()