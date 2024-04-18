[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10991918.svg)](https://zenodo.org/records/10991918)

# TurboPE-BNS

Repository with scripts used for the paper [Robust parameter estimation within minutes on gravitational wave signals from binary neutron star inspirals](https://arxiv.org/abs/2404.11397). Please raise an issue on this Github repository if you encounter any issue. If you make use of the methods presented in this work, please cite:
```bibtex
@misc{wouters2024robust,
      title={Robust parameter estimation within minutes on gravitational wave signals from binary neutron star inspirals}, 
      author={Thibeau Wouters and Peter T. H. Pang and Tim Dietrich and Chris Van Den Broeck},
      year={2024},
      eprint={2404.11397},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM}
}
```

Also make sure to cite the original `Jim` paper:

```bibtex
@article{Wong:2023lgb,
    author = "Wong, Kaze W. K. and Isi, Maximiliano and Edwards, Thomas D. P.",
    title = "{Fast Gravitational-wave Parameter Estimation without Compromises}",
    eprint = "2302.05333",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.3847/1538-4357/acf5cd",
    journal = "Astrophys. J.",
    volume = "958",
    number = "2",
    pages = "129",
    year = "2023"
}
```

## Reproducibility

The aim of this repository is to allow other users to reproduce the results shown in the paper and to enable them to easily get started with BNS parameter estimation with `Jim`. Therefore, we mention explicitly which versions of `ripple`, `flowMC` and `Jim` were used for the runs mentioned in this work. Below, we provide links to the repositories' version at the time of writing. We also provide a Zenodo release, which can be found [here](https://zenodo.org/records/10991918) with ZIP files containing all three packages, as well as the version of this repository at the time of releasing the arXiv submission

- `ripple`: [link](https://github.com/ThibeauWouters/flowMC/tree/84cdf3847d1fb2df8fc996086381d90a446c1ac2)
- `flowMC`: [link](https://github.com/ThibeauWouters/flowMC/tree/84cdf3847d1fb2df8fc996086381d90a446c1ac2)
- `jim`: [link](https://github.com/ThibeauWouters/jim/commit/a35403ebeb9a1de8d68c17d0c390b58afc5f51f9)

## Overview of this repository

We provide a quick overview of the contents of this repository.

### RB

This directory contains the output files for the relative binning with Bilby runs mentioned in Table IV. 

### ROQ

This directory contains the output files for the ROQ with Bilby runs mentioned in Table IV. 

### ROQ-build

An attempt at constructing our own ROQ bases for the estimate of Appendix C. Eventually, we used the estimates provided by Soichiro Morisaki for Appendix C, as we find this source to be more trustworthy.

### Data

A few data files used in the postprocessing scripts or injection runs. 

### Figures

Final figures used in the paper.

### Injections

Directory storing all the scripts related to the simulated events, as well as the `outdir` directories of all runs. The produced posterior samples are too large to be stored on Github and can be shared upon request. 

### Postprocessing

A few scripts used to create the plots for the paper.

### real_events

All scripts used in the analysis of the real events, GW170817 and GW190245, and the respective `outdirs`. The produced posterior samples are too large to be stored on Github and can be shared upon request. 

Datafiles for the real events are too large to be stored on Github and can be found on the respective GWOSC websites. If more information is needed, please raise an issue on this Github.
- [GW190425](https://gwosc.org/eventapi/html/O3_Discovery_Papers/GW190425/v1/)
- [GW170817](https://gwosc.org/events/GW170817/)