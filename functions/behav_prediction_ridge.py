from sklearn.linear_model import Ridge,RidgeCV,LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from scipy.stats import spearmanr
import numpy as np
import os
import pickle
import h5py
import sys
sys.path.append('/projects/f_mc1689_1/ClinicalActFlow/docs/scripts/')
from slurmUtils import *

def behav_prediction(X,y,X2=False,iterations=200,folds=5,normalise=False,ridge=True):
    '''
    Takes brain data (X) and to be predicted behavioural data (y) and performs
    _iteration_ number of CVs of _folds_ number of folds (default=5) cross validation
    
    Inputs:
        X - brain data (subj x roi)
        y - behaviour (subj)
        X2 - if you want to test on other data (e.g., real or actflow activations).
            Needs to be the same size as X (def=False)
        iterations - number of outer iterations (def=200)
        folds - number of folds in cv (def=5)
        normalise - whether to zscore the X data. Note that this is done within the training
                set and then applied to the testing set.
        verbose - print the result of each fold (def=False)
    
    Outputs
        A dict with:
            r - correlation between real and predicted
            MSE - mean squared error
            MedSE - median squared error
            Rsqr - R^2
        For each permutation.
    '''
    alphas = (0.000001,0.000001,0.00001,0.0001,0.001,0.01,0.1, 1, 10,100,1000,10000)
    
    #output lists
    r_all = []
    MAE_all = []
    Rsqr_all = []
    betas_all = []
    y_pred_all = []
    
    for i in range(iterations):
        cv = KFold(n_splits=folds,shuffle=True)
        
        #preallocation
        r    = []
        MAE = []
        Rsqr = []
        betas = []
        y_pred_out = np.zeros(np.shape(y))
        
        for train_index, test_index in cv.split(X):
            #organise the data into training and testing sets
            X_train = X[train_index,:].copy()
            y_train = y[train_index].copy()
            
            # allocate the test data from the same dataset (validation fold)
            # or from the X2 dataset (for testing outside the original data)
            if X2 is False:
                #print('\t Using X data')
                X_test  = X[test_index,:].copy()
            elif X2 is not False:
                #print('\t Using X2 data')
                X_test = X2[test_index,:].copy()
                
            if normalise:
                m = np.mean(X_train,axis=0)
                sd = np.std(X_train,axis=0)
                # Apply training normalisation to training and testing data
                X_train = ((X_train - m)/sd)
                X_test = ((X_test - m)/sd)
            
            if ridge==True:
                # find the ridge parameter via built in LOO (ridgeCV) [ONLY in the training data]
                alpha_model = RidgeCV(alphas=alphas, fit_intercept=True)
                alpha_model.fit(X_train,y_train)

                # use the alpha parameter
                reg_model = Ridge(alpha=alpha_model.alpha_,fit_intercept=True)
            else:
                reg_model = LinearRegression(fit_intercept=True)
                
            # fit the training model
            reg_model.fit(X_train,y_train)
            
            # Make predictions in held out fold
            y_pred = reg_model.predict(X_test)
            
            # test predictions
            #r.append(np.corrcoef(y[test_index],y_pred)[0,1])
            r.append(spearmanr(y[test_index],y_pred)[0])
            MAE.append(mean_absolute_error(y[test_index],y_pred))
            Rsqr.append(r2_score(y[test_index],y_pred))
            
            # save the betas and predictions for this model
            betas.append(reg_model.coef_)
            y_pred_out[test_index] = y_pred.copy()
            
        r_all.append(np.mean(r))
        MAE_all.append(np.mean(MAE))
        Rsqr_all.append(np.mean(Rsqr))
        betas_all.append(np.mean(betas,axis=0))
        y_pred_all.append(y_pred_out)
    
    # output the averages across iterations
    output= {
        'r': np.mean(r_all),
        'MAE': np.mean(MAE_all),
        'Rsqr': np.mean(Rsqr_all),
        'betas':np.mean(betas_all,axis=0),
        'y_pred':np.mean(y_pred_all,axis=0)}
    
    # output the averages across iterations
    output_all= {
        'r': r_all,
        'MAE': MAE_all,
        'Rsqr': Rsqr_all,
        'betas':betas_all,
        'y_pred':y_pred_all}

    return output,output_all

def py_to_slurm_permutation(input_file,output_folder,perms_low,perms_high,file_name,python_script,partition ='nm3',time = '00:10:00',nnodes=1,ncpus=2,mem=4000,submit=True,suppress_output=True):
    
    # hard coded directory
    path = '/projects/f_mc1689_1/ClinicalActFlow/docs/scripts/slurmUtils/'
    
    # open the bash file
    bash_file = path + 'batchScripts/' + file_name + '_' + perms_low + '-' + perms_high + '.sh'
    file_slurm = open(bash_file, 'w')
    
    # slurm parameters
    file_slurm.write('#!/bin/bash\n')
    file_slurm.write('#SBATCH --partition=' + partition + '\n')
    file_slurm.write('#SBATCH --job-name=' + file_name + '_' + perms_low + '-' + perms_high + '\n')
    file_slurm.write('#SBATCH --requeue\n')
    file_slurm.write('#SBATCH --time=' + time + '\n')
    file_slurm.write('#SBATCH --nodes=' + str(nnodes) + '\n')
    file_slurm.write('#SBATCH --ntasks=1\n')
    file_slurm.write('#SBATCH --cpus-per-task=' + str(ncpus) + '\n')
    file_slurm.write('#SBATCH --mem-per-task=' + str(mem) + '\n')
    if suppress_output is False:
        file_slurm.write('#SBATCH --export=ALL\n')
        file_slurm.write('#SBATCH --output=' + path + '/batchScripts/slurm.' + file_name + '_' + perms_low + '-' + perms_high + '.out\n')
        file_slurm.write('#SBATCH --error=' + path + '/batchScripts/slurm.' + file_name + '_' + perms_low + '-' + perms_high + '.err\n')
    #instructions
    #file_slurm.write('module load python\n')
    file_slurm.write('python ' + path + python_script + " '" + input_file + "' " + " '" + output_folder + "' " + " '" + perms_low + "' " + " '" + perms_high + "' " + '\n')
    file_slurm.close()
    
    # run the bash scripts
    os.system("chmod 755 " + bash_file)
    if submit is True:
        os.system("sbatch " + bash_file)
    if suppress_output is False:
        print('\tjob submitted: ',bash_file)
        
def behav_prediction_permutation_wrapper(X,y,X2,load=True,permutations=1000,iterations=200,n_chunks=10,folds=5,time_per_perm=20,mem=4000,label='/projects/f_mc1689_1/ClinicalActFlow/data/results/CV_permutations/scap_accuracy'):
    '''
    A wrapper function that sends behavioural prediction permutations to SLURM
    '''
    chunk_size = int(permutations / n_chunks)

    # based on testing
    time = int(np.ceil(time_per_perm * chunk_size/60))
    time='00:'+str(time).zfill(2)+':00'

    # h5 path for the results:
    path = label + '_' + str(permutations) + '_' + str(iterations) + '_' + str(folds)

    if load is True:
        print('Loading results:')
        # create dict for all results
        permutation_results = {}
        permutation_results['r'] = []
        permutation_results['MAE'] = []
        permutation_results['Rsqr'] = []
        permutation_results['betas'] = []
        permutation_results['y_pred'] = []

        # load the pickleso
        for i in range(permutations):
            with open(path+'/perm' + str(i) + '.pickle', 'rb') as f:
                data = pickle.load(f)
                permutation_results['r'].append(data['r'])
                permutation_results['MAE'].append(data['MAE'])
                permutation_results['Rsqr'].append(data['Rsqr'])
                permutation_results['betas'].append(data['betas'])
                permutation_results['y_pred'].append(data['y_pred'])
        print('Results loaded!')
        return permutation_results

    else:
        print('Running analysis:')
        print('\tallotted time per node=',time,'minutes')
        print('\tgenerating and saving shuffled data')
        
        # create a folder for the results
        os.makedirs(path, exist_ok=True)
        
        # split the permutations and call py_to_slurm
        chunk=0
        for i in np.arange(0,permutations,chunk_size):
            # define variables for the current chunk
            perms_low = i.copy()
            perms_high = i + chunk_size
            current_perms = np.arange(perms_low,perms_high)
            n_current_perms = len(current_perms)
            
            # organise shuffled data
            y_pool = []
            for p in current_perms:
                y_pool.append(shuffle(y))

            X_pool = [X] * n_current_perms
            if X2 is False:
                X2_pool = [False] * n_current_perms
            else:
                X2_pool = [X2] * n_current_perms
            iterations_pool = iterations
            folds_pool = folds
            normalise_pool = False

            # save as a h5py file
            file = label + '_' + str(permutations) + '_' + str(iterations) + '_' + str(folds) + '_' + str(chunk) + '.h5'
            h5f = h5py.File(file,'a')

            outname = '/inputs'
            try:
                h5f.create_dataset(outname + '/X',data=X_pool,compression='gzip', compression_opts=9)
                h5f.create_dataset(outname + '/X2',data=X2_pool,compression='gzip', compression_opts=9)
                h5f.create_dataset(outname + '/y',data=y_pool,compression='gzip', compression_opts=9)
                h5f.create_dataset(outname + '/iterations',data=iterations_pool)
                h5f.create_dataset(outname + '/folds',data=folds_pool)
                h5f.create_dataset(outname + '/normalise',data=normalise_pool)
            except:
                del h5f[outname]
                h5f.create_dataset(outname + '/X',data=X_pool,compression='gzip', compression_opts=9)
                h5f.create_dataset(outname + '/X2',data=X2_pool,compression='gzip', compression_opts=9)
                h5f.create_dataset(outname + '/y',data=y_pool,compression='gzip', compression_opts=9)
                h5f.create_dataset(outname + '/iterations',data=iterations_pool)
                h5f.create_dataset(outname + '/folds',data=folds_pool)
                h5f.create_dataset(outname + '/normalise',data=normalise_pool)
            h5f.close()

            py_to_slurm_permutation(input_file=file,
                                    output_folder = path,
                                    perms_low=str(perms_low),
                                    perms_high=str(perms_high),
                                    file_name='beh-pred',
                                    python_script='behav_prediction_slurm.py',
                                    time=time,
                                    mem=mem,
                                    submit=True,
                                    suppress_output=False)
            chunk = chunk+1