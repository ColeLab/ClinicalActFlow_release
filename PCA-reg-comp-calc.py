import h5py
import numpy as np
import pandas as pd
import pickle
import time
from scipy import signal
import sys
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

## global variables
# parcellation
PARC = 'cabn'

# out directories
results_dir = '/projects/f_mc1689_1/ClinicalActFlow/data/results/PCA-reg/'
figure_dir = '/projects/f_mc1689_1/ClinicalActFlow/docs/figures/PCA-reg/'


PARC = 'cabn'
OUTPUT_DIR = '/projects/f_mc1689_1/ClinicalActFlow/data/prepro/Output_'+PARC+'/'
task_list = ['rest','taskswitch','bart','stopsignal']
model_task ='24pXaCompCorXVolterra'
model_rest = '24pXaCompCorXVolterra-spikeReg'
zscore=False

def get_multi_timeseries(subj):


    h5f = h5py.File(OUTPUT_DIR + subj + '_GLMOutput.h5','r')
    data = np.zeros((718,1))
    
    for task in task_list:

        # get the right nuisance model
        if task == 'rest':
            model = model_rest
        else:
            model = model_task

        # load the data
        if task == 'rest':
            # load residuals from specific parcellation
            current_data = h5f[task]['nuisanceReg_resid_'+model][:].copy()
        else:
            # get FIR data residuals
            current_data = h5f[task]['FIR']['taskActivity_resid_'+model][:].copy()

        # demean data
        current_data = signal.detrend(current_data,axis=0,type='constant')

        # zscore data
        if zscore:
            current_data = stats.zscore(current_data,axis=0)

        # concatenate data
        data = np.hstack((data,current_data.T))
    h5f.close()

    # remove the initial zero col
    data = np.delete(data,0,axis=1)
    
    return data

# this function is slightly altered compared to the activity flow toolbox
# I wanted to return the variances from the PCA as well...



def pc_multregconn_var(activity_matrix, target_ts=None, n_components=None, n_comp_search=False, n_components_min=1, n_components_max=None, parcelstoexclude_bytarget=None):
    """
    activity_matrix:    Activity matrix should be nodes X time
    target_ts:             Optional, used when only a single target time series (returns 1 X nnodes matrix)
    n_components:  Optional. Number of PCA components to use. If None, the smaller of number of nodes or number of time points (minus 1) will be selected.
    n_comp_search: Optional. Boolean indicating whether to search for the best number of components based on cross-validation generalization (to reduce overfitting).
    n_components_min: Optional. The smallest number to test in the n_comp_search.
    n_components_max: Optional. The largest number to test in the n_comp_search.
    parcelstoexclude_bytarget: Optional. A dictionary of lists, each listing parcels to exclude for each target parcel (e.g., to reduce potential circularity by removing parcels near the target parcel). Note: This is not used if target_ts is set.
    
    Output: connectivity_mat (formatted targets X sources), n_components
    """
    tot_variance_bynode = []
    nnodes = activity_matrix.shape[0]
    timepoints = activity_matrix.shape[1]
    
    if n_components == None:
        n_components = np.min([nnodes-1, timepoints-1])
    else:
        if nnodes<n_components or timepoints<n_components:
            print('activity_matrix shape: ',np.shape(activity_matrix))
            raise Exception('More components than nodes and/or timepoints! Use fewer components')        
    
    #De-mean time series
    activity_matrix_mean = np.mean(activity_matrix,axis=1)
    activity_matrix = activity_matrix - activity_matrix_mean[:, np.newaxis]
    
    if target_ts is None:
        
        #Cross-validation to find optimal number of components (based on mean MSE across all nodes)
        if n_comp_search:
            if n_components_max is None:
                n_components_max = np.min([nnodes-1, timepoints-1])
            componentnum_set=np.arange(n_components_min,n_components_max+1)
            mse_regionbycomp = np.zeros([np.shape(componentnum_set)[0],nnodes])
            for targetnode in range(nnodes):
                othernodes = list(range(nnodes))
                #Remove parcelstoexclude_bytarget parcels (if flagged); parcelstoexclude_bytarget is by index (not parcel value)
                if parcelstoexclude_bytarget is not None:
                    parcelstoexclude_thisnode=parcelstoexclude_bytarget[targetnode].tolist()
                    parcelstoexclude_thisnode.append(targetnode) # Remove target node from 'other nodes'
                    othernodes = list(set(othernodes).difference(set(parcelstoexclude_thisnode)))
                else:
                    othernodes.remove(targetnode) # Remove target node from 'other nodes'
                X = activity_matrix[othernodes,:].T
                y = activity_matrix[targetnode,:]
                #Run PCA
                pca = PCA()
                Xreg_allPCs = pca.fit_transform(X)
                mscv_vals=np.zeros(np.shape(componentnum_set)[0])
                comp_count=0
                for comp_num in componentnum_set:
                    regr = LinearRegression()
                    Xreg = Xreg_allPCs[:,:comp_num]
                    regr.fit(Xreg, y)
                    # Cross-validation
                    y_cv = cross_val_predict(regr, Xreg, y, cv=10)
                    mscv_vals[comp_count] = mean_squared_error(y, y_cv)
                    comp_count=comp_count+1
                mse_regionbycomp[:,targetnode] = mscv_vals
            min_comps_means = np.mean(mse_regionbycomp, axis=1)
            n_components=componentnum_set[np.where(min_comps_means==np.min(min_comps_means))[0][0]]
            print('n_components = ' + str(n_components))
        
        connectivity_mat = np.zeros((nnodes,nnodes))
        for targetnode in range(nnodes):
            othernodes = list(range(nnodes))
            #Remove parcelstoexclude_bytarget parcels (if flagged); parcelstoexclude_bytarget is by index (not parcel value)
            if parcelstoexclude_bytarget is not None:
                parcelstoexclude_thisnode=parcelstoexclude_bytarget[targetnode].tolist()
                parcelstoexclude_thisnode.append(targetnode) # Remove target node from 'other nodes'
                othernodes = list(set(othernodes).difference(set(parcelstoexclude_thisnode)))
            else:
                othernodes.remove(targetnode) # Remove target node from 'other nodes'
            X = activity_matrix[othernodes,:].T
            y = activity_matrix[targetnode,:]
            #Run PCA on source time series
            pca = PCA(n_components)
            reduced_mat = pca.fit_transform(X) # Time X Features
            components = pca.components_
            #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
            regrmodel = LinearRegression()
            reg = regrmodel.fit(reduced_mat, y)
            #Convert regression betas from component space to node space
            betasPCR = pca.inverse_transform(reg.coef_)
            connectivity_mat[targetnode,othernodes]=betasPCR
            
            # ljh add
            variance = pca.explained_variance_ratio_
            tot_variance = np.sum(variance)
            tot_variance_bynode.append(tot_variance)
    else:
        #Remove time series mean
        target_ts = target_ts - np.mean(target_ts)
        #Computing values for a single target node
        connectivity_mat = np.zeros((nnodes,1))
        X = activity_matrix.T
        y = target_ts
        #Cross-validation to find optimal number of components
        if n_comp_search:
            componentnum_set=np.arange(n_components_min,n_components_max+1)
            mscv_vals=np.zeros(np.shape(componentnum_set)[0])
            comp_count=0
            for comp_num in componentnum_set:
                mscv_vals[comp_count] = pcr_cvtest(X,y, pc=comp_num, cv=10)
                comp_count=comp_count+1
            n_components=componentnum_set[np.where(mscv_vals==np.min(mscv_vals))[0][0]]
        #Run PCA on source time series
        pca = PCA(n_components)
        reduced_mat = pca.fit_transform(X) # Time X Features
        components = pca.components_
        #Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
        reg = LinearRegression().fit(reduced_mat, y)
        #Convert regression betas from component space to node space
        betasPCR = pca.inverse_transform(reg.coef_)
        connectivity_mat=betasPCR

    return connectivity_mat,tot_variance_bynode

if __name__ == '__main__':
    
    # get subject name
    subj = sys.argv[1]
    
    # variables
    component_list = [50,100,150,200,250,300]
    
    # pre allocate
    fc = np.zeros((718,718,len(component_list)))
    tot_variance_bynode = np.zeros((718,len(component_list)))
    for c,comp in enumerate(component_list):
        print('comp=',comp)
        start = time.time()
        data = get_multi_timeseries(subj)
        print(data.shape)
        fc[:,:,c],tot_variance_bynode[:,c] = pc_multregconn_var(data,n_components=comp)
        print('\t',np.round((time.time() - start)/60),'minutes')

    # save data
    h5f = h5py.File(results_dir + subj + 'PCA-reg.h5','a')

    try:
        h5f.create_dataset('/fc',data=fc)
        h5f.create_dataset('/variance',data=tot_variance_bynode)
    except:
        del h5f['/fc'],h5f['/variance']
        h5f.create_dataset('/fc',data=fc)
        h5f.create_dataset('/variance',data=tot_variance_bynode)
    h5f.close()