import numpy as np
import json
from scipy.stats import percentileofscore, uniform, kstest
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt

outdir = '/home/thibeau.wouters/TurboPE-BNS/injections/outdir_NRTv2/'
problematic_injections = [1, 2, 45, 74, 81, 89]
#problematic_injections = [1, 2]

naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

one_sided_pvalue = {}
two_sided_pvalue = {}
for param_idx, param_name in enumerate(naming):
    one_sided_pvalue[param_name] = []
    two_sided_pvalue[param_name] = []
    
    for i in range(1, 104):
        # if i in problematic_injections:
        #     continue
        # fetch the injected value
        with open(f'{outdir}/injection_{i}/config.json') as f:
            config = json.load(f)
            tc_inj = config[param_name]
        # fetch the posterior samples
        posterior_samples = np.load(f'{outdir}/injection_{i}/results_production.npz')
        tc_samples = posterior_samples['chains'].T[param_idx].flatten()

        p_value = percentileofscore(tc_samples, tc_inj) / 100.
        one_sided_pvalue[param_name].append(p_value)
        
        two_sided_p = 1 - 2 * min(p_value, 1 - p_value)
        two_sided_pvalue[param_name].append(two_sided_p)
        

    one_sided_pvalue[param_name] = np.array(one_sided_pvalue[param_name])
    two_sided_pvalue[param_name] = np.array(two_sided_pvalue[param_name])

plt.figure(1)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], color='k', linestyle='--')
plt.xlabel('Bayesian credible interval')
plt.ylabel('Fraction of events')
plt.hist(one_sided_pvalue, bins=1000, density=True, histtype='step', cumulative=1)
plt.savefig('tc_cumhist.pdf', bbox_inches='tight')

# calculate the p-value
for key, value in one_sided_pvalue.items():
    
    stat, p_value = kstest(value, uniform.cdf)
    print(f"{key}: The p-value is {p_value}")

    # stat, p_value = kstest(two_sided_pvalue, uniform.cdf)
    # print(f"The p-value is {p_value}")