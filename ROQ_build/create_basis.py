from itertools import product
import numpy as np
import optparse

from gwrom.cbc.generator import aligned_spin_grid, aligned_spin_random
from gwrom.cbc.waveform import MBFDWaveform
from gwrom.cbc.calibration import calibration_random
from gwrom.core.roq import construct_basis
from gwrom.core.utils import combine_params, load_results

import time
start = time.time()

print("Starting create basis now!")

# parse commands
parser = optparse.OptionParser()
parser.add_option("--mcmin", dest="mcmin", type="float")
parser.add_option("--mcmax", dest="mcmax", type="float")
parser.add_option("--reference-chirp-mass", dest="reference_chirp_mass", type="float")
parser.add_option("--duration", dest="duration", type="float")
parser.add_option("--fhigh", dest="fhigh", type="float")
parser.add_option("--quadratic", action="store_true", default=False, dest="quadratic")
parser.add_option("--amplitude-max", dest="amplitude_max", type="float")
parser.add_option("--phase-max", dest="phase_max", type="float")
parser.add_option("--extrapolate", action="store_true", default=False, dest="extrapolate")
parser.add_option("--input", dest="input", action="append", type="string")
parser.add_option("--out", dest="out", type="string")
parser.add_option("--torelance", dest="torelance", default=1e-10, type="float")
parser.add_option("--validation-num", dest="validation_num", default=int(3 * 1e4), type="int")
parser.add_option("--nprocess", dest="nprocess", default=1, type="int")
(options, args) = parser.parse_args()

# construct waveform object
flow, fhigh = 20., options.fhigh
duration = options.duration
approximant = "IMRPhenomPv2_NRTidalv2"
fref = 20.
reference_chirp_mass = options.reference_chirp_mass
n_nodes = 10
if options.extrapolate:
    fill_value = "extrapolate"
else:
    fill_value = 0.
wf = MBFDWaveform(
    flow=flow, fhigh=fhigh, approximant=approximant, duration=duration,
    fref=fref, reference_chirp_mass=reference_chirp_mass, quadratic=options.quadratic,
    add_calibration=True, n_nodes=n_nodes, fill_value=fill_value
)

# parameter range
mcmin, mcmax = options.mcmin, options.mcmax
qmin, qmax = 1. / 8., 1.
a1zmin, a1zmax = -0.05, 0.05
a2zmin, a2zmax = -0.05, 0.05

# function to add calibration parameters
def add_calibration_params(params):
    n_sample = len(params)
    kwargs = {}
    for i in range(n_nodes):
        kwargs[f"amplitude_{i}_min"] = -options.amplitude_max
        kwargs[f"amplitude_{i}_max"] = options.amplitude_max
        kwargs[f"phase_{i}_min"] = -options.phase_max
        kwargs[f"phase_{i}_max"] = options.phase_max
    params_calib = calibration_random(n_sample, n_nodes, **kwargs)
    for p1, p2 in zip(params, params_calib):
        p1.update(p2)

def add_extreme_calibration_params(params):
    extreme_calib_params = []
    for s1, s2 in product([-1, 1], [-1, 1]):
        p = {}
        for i in range(n_nodes):
            p[f"amplitude_{i}"] = options.amplitude_max * (-1)**i * s1
            p[f"phase_{i}"] = options.phase_max * (-1)**i * s2
        extreme_calib_params.append(p)
    n_calib = len(extreme_calib_params)
    for i, p in enumerate(params):
        p.update(extreme_calib_params[i % n_calib])

def add_zero_calibration_params(params):
    zero_calib = {}
    for i in range(n_nodes):
        zero_calib[f"amplitude_{i}"] = 0.
        zero_calib[f"phase_{i}"] = 0.
    for p in params:
        p.update(zero_calib)

# initial training grid
mcnum, qnum, a1znum, a2znum = 5, 5, 5, 5
## zero calibration params
training_set = aligned_spin_grid(
    mcmin, mcmax, mcnum, qmin, qmax, qnum,
    a1zmin, a1zmax, a1znum, a2zmin, a2zmax, a2znum
)
add_zero_calibration_params(training_set)
## add extreme calibration params
to_add = aligned_spin_grid(
    mcmin, mcmax, mcnum, qmin, qmax, qnum,
    a1zmin, a1zmax, a1znum, a2zmin, a2zmax, a2znum
)
add_extreme_calibration_params(to_add)
training_set = combine_params(training_set, to_add)
## add random calibration params
to_add = aligned_spin_grid(
    mcmin, mcmax, mcnum, qmin, qmax, qnum,
    a1zmin, a1zmax, a1znum, a2zmin, a2zmax, a2znum
)
add_calibration_params(to_add)
training_set = combine_params(training_set, to_add)
## add input params
if options.input is not None:
    for i in options.input:
        to_add = load_results(i)["selected_params"]
        training_set = combine_params(training_set, to_add)

# set up random parameter generator
def generator(n):
    n_zero = n // 3
    n_extreme = n // 3
    n_random = n - n_zero - n_extreme
    ## zero calibration params
    params = aligned_spin_random(
        mcmin, mcmax, qmin, qmax, a1zmin, a1zmax, a2zmin, a2zmax, n_zero
    )
    add_zero_calibration_params(params)
    ## add extreme calibration params
    to_add = aligned_spin_random(
        mcmin, mcmax, qmin, qmax, a1zmin, a1zmax, a2zmin, a2zmax, n_extreme
    )
    add_extreme_calibration_params(to_add)
    params = combine_params(params, to_add)
    ## add random calibration params
    to_add = aligned_spin_random(
        mcmin, mcmax, qmin, qmax, a1zmin, a1zmax, a2zmin, a2zmax, n_random
    )
    add_calibration_params(to_add)
    params = combine_params(params, to_add)
    return params

# construct basis.
validation_num = options.validation_num
torelance_greedy = options.torelance
torelance_validation = options.torelance
if options.quadratic:
    torelance_greedy /= len(wf.data_points)
    torelance_validation /= len(wf.data_points)
construct_basis(
    wf, training_set, generator, validation_num, torelance_greedy,
    torelance_validation, options.out, multiband_greedy=False,
    multiband_validation=False, validate_interpolant=False, nprocess=options.nprocess,
    max_add=5000
)

end = time.time()

print("HEY THERE, total time was:")
print(end- start)
