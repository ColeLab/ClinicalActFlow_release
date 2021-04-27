import numpy as np
from sklearn.utils import shuffle
from scipy.stats import ttest_ind

# Code to perform permutation testing to control for family-wise error (FWE)
# Using max-T approach as described in Nichols & Holmes (2002)
# Nichols TE, Holmes AP. (2002). Nonparametric permutation tests for functional neuroimaging: A primer with Examples. Hum. Brain Mapp., 15: 1-25. doi:10.1002/hbm.1058

# this is a between- groups approach as opposed to the other examples available in the ColeLab repo
# see https://github.com/ColeLab/MultipleComparisonsPermutationTesting/blob/master/pythonCode/permutationTesting.py

# Code uses a similar framework to: https://www.mathworks.com/matlabcentral/fileexchange/54585-mult_comp_perm_t2-data1-data2-n_perm-tail-alpha_level-mu-t_stat-reports-seed_state?s_tid=prof_contriblnk

def ttest_ind_FWE(data1,data2,n_perm=5000,alpha_level=0.05,equal_var=True,random_seed=None):

    '''
    Max-t non-parametric permutation based method for FWE control of multiple comparisons.
    This function is appropriate for indepedent group comparisons i.e., scipy.stats.ttest_ind
    Tests for two-tailed group differences

    INPUTS (REQUIRED)
    data1   - 2D matrix of data (Observation x Variable)
    data2   - 2D matrix of data (Observation x Variable)

    INPUTS (OPTIONAL)
    n_perm - Number of random permutations used to estimate the distribution of the null hypothesis.  
            Manly (1997) suggests using at least 1000 permutations for an alpha level of 0.05 and at 
            least 5000 permutations for an alpha level of 0.01. Default = 5000

    alpha_level - Desired family-wise alpha level. Note, because of the finite number of possible 
            permutations, the exact desired family-wise alpha may not be possible. Thus, the 
            closest approximation is used and output as est_alpha. Default = 0.05

    equal_var - Whether to use Welch's t-test (False) or not (True). See documentation for
            stats.scipy.ttest_ind. Default = True

    random_state - The initial state of the random number generating stream. If you pass a value 
            from a previous run of this function, it should reproduce the exact same values. Default = None

    OUTPUTS
    t_orig - t values
    p _adj - adjusted p-values
    '''
    
    # set random seed
    np.random.seed(random_seed)
    
    #organise data
    all_data = np.vstack((data1,data2))
    n_obs1 = np.shape(data1)[0]
    n_obs2 = np.shape(data2)[0]
    totalObs = n_obs1 + n_obs2

    if data1.ndim==1:
        n_var1 = 1
    else:
        n_var1 = np.shape(data1)[1]

    if data2.ndim==1:
        n_var2 = 1
    else:
        n_var2 = np.shape(data2)[1]

    print('Observations in g1:',n_obs1,'| Variables in g1:',n_var1)
    print('Observations in g2:',n_obs2,'| Variables in g2:',n_var2)
    if n_var1!=n_var2:
        raise Exception('The number of variables in data1 and data2 are not equal.')

    # degrees of freedom
    df = totalObs - 2;

    # Compute permutations
    print('Starting permutations...')
    max_t = np.zeros((n_perm))
    for perm in range(n_perm):

        # randomly assign participants to conditions
        r = shuffle(range(totalObs))
        grp1 = r[:n_obs1]
        grp2 = r[n_obs1::]

        # compute most extreme t-score
        x1 = all_data[grp1,:]
        x2 = all_data[grp2,:]

        # standard t-test using scipy.stats.ttest_ind
        t, p = ttest_ind(x1, x2,equal_var=equal_var)
        max_t[perm] = np.max(t)
    print('\tFinished permutations...')

    # Compute critical t's
    crit_t = np.array([0,0],dtype=float)
    crit_t[1] = np.percentile(max_t,100-100*alpha_level)
    crit_t[0] = crit_t[1] * -1
    est_alpha = np.mean(max_t >= crit_t[1])
    print('Desired Alpha level:',alpha_level,'| Estimated FWE alpha:',est_alpha)

    # Compute statistic in real data
    t_orig, _ = ttest_ind(data1, data2,equal_var=equal_var)

    # Compute adjusted p-values
    p_adj= np.zeros(n_var1)
    for t in range(n_var1):
        p_adj[t] = np.mean(max_t >= abs(t_orig[t])) #note mx_t are now all positive due to abs command above
        
    return t_orig,p_adj