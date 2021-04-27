from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr,pearsonr,ttest_1samp,zscore
import numpy as np
from .mean_absolute_perc_error import *

def actflow_tests(actVect_group,actPred_group,normalise=False):
    '''
    Calculates basic accuracy metrics for comparing real and predicted activations.
    Very similar to actflowtest.py from the actflowtoolbox
    Activations should be shaped (node,condition,subject)
    '''
    if normalise:
        actVect_group = zscore(actVect_group,axis=0)
        actPred_group = zscore(actPred_group,axis=0)
    n_nodes = np.shape(actVect_group)[0]
    n_subs = np.shape(actVect_group)[1]

    r   = np.zeros((n_subs))
    rs  = r.copy()
    MAE = r.copy()
    MAPE = r.copy()
    Rsqr= r.copy()

    # Calculate model accuracy at the whole brain level for each condition in the task
    for subj in range(n_subs):
        real = actVect_group[:,subj]
        pred = actPred_group[:,subj]

        ## Metrics
        # correlation between real and predicted
        r[subj] = pearsonr(real,pred)[0]
        # rs correlation between real and predicted
        rs[subj] = spearmanr(real,pred)[0]
        # MAE between real and predicted
        MAE[subj] = mean_absolute_error(real,pred)
        # MAE between real and predicted
        MAPE[subj] = mean_absolute_perc_error(real,pred)
        # R^2 between real and predicted
        Rsqr[subj] = r2_score(real,pred)

    # Test if metrics are different from zero (grand mean across conditions) 
    r_stat = ttest_1samp(r,0.0)
    rs_stat = ttest_1samp(rs,0.0)
    Rsqr_stat = ttest_1samp(Rsqr,0.0)

    # print averages
    print('Mean r across subjs:',np.round(np.mean(r,axis=0),3), '|1samp t:',np.round(r_stat[0],2),'p:',np.round(r_stat[1],5))
    print('Mean MAE  across subjs:',np.round(np.mean(MAE,axis=0),3))
    print('Mean MAPE  across subjs:',np.round(np.mean(MAPE,axis=0),3))
    print('Mean R^2  across subjs:',np.round(np.mean(Rsqr,axis=0),3), '|1samp t:',np.round(Rsqr_stat[0],2),'p:',np.round(Rsqr_stat[1],5))
    return r,rs,MAE,MAPE,Rsqr

def actflowcalc_hold_out_roi(actVect, fcMat, held_out_roi=None,roi_list=None):
    """
    - Zeros out 'held_out_roi' in actVect so that they cannot contribute to any predictions.
    - Also has the option of only completed the activity flow calculation on a subset of rois 
    (this speeds up the computation)
    - Otherwise is very similiar to the code within the ActFlow toolbox (actflowcalc.py).
    
    INPUTS:
        actVect: node vector with activation values
        fcMat: node x node matrix with connectiivty values
        held_out_roi: the regions to hold out completely from ALL calculations. (In addition 
        to the target roi)
        roi_list: the subset of regions to perform the analysis on 
    """
    
    # zero out held_out_roi
    if held_out_roi != None:
        actVect[held_out_roi] = 0
    
    numRegions=np.shape(actVect)[0]
    actPredVector=np.zeros((numRegions,))
    actWeights = np.zeros((numRegions,numRegions))
    
    if roi_list is None:
        for heldOutRegion in range(numRegions):
            otherRegions=list(range(numRegions))
            actPredVector[heldOutRegion]=np.sum(actVect[otherRegions]*fcMat[heldOutRegion,otherRegions])
            # don't sum the weights and save out
            actWeights[heldOutRegion,otherRegions] = actVect[otherRegions]*fcMat[heldOutRegion,otherRegions]
    else:
        # if user has provided a list - only loop through those rois
        for heldOutRegion in roi_list:
            otherRegions=list(range(numRegions))
            otherRegions.remove(heldOutRegion)
            actPredVector[heldOutRegion]=np.sum(actVect[otherRegions]*fcMat[heldOutRegion,otherRegions])
            # don't sum the weights and save out
            actWeights[heldOutRegion,otherRegions] = actVect[otherRegions]*fcMat[heldOutRegion,otherRegions]
    return actPredVector,actWeights

def roi_level_accuracy(activity,predicted_activity,roi_list):
    
    r = np.zeros((len(roi_list)))
    MAE = r.copy()
    MAPE = r.copy()
    Rsqr = r.copy()

    for i,roi in enumerate(roi_list):

#             # do the contrast & stack participants
#             real = np.hstack((np.mean(activity['scap']['CTRL'][:,6::,:],axis=1) - np.mean(activity['scap']['CTRL'][:,0:6,:],axis=1),
#                               np.mean(activity['scap']['SCHZ'][:,6::,:],axis=1) - np.mean(activity['scap']['SCHZ'][:,0:6,:],axis=1)))

#             pred = np.hstack((np.mean(predicted_activity['scap']['CTRL'][:,6::,:],axis=1) - np.mean(predicted_activity['scap']['CTRL'][:,0:6,:],axis=1),
#                               np.mean(predicted_activity['scap']['SCHZ'][:,6::,:],axis=1) - np.mean(predicted_activity['scap']['SCHZ'][:,0:6,:],axis=1)))
            
            # do the contrast & stack participants
            real = np.hstack((activity['scap']['CTRL'][:,1,:] - activity['scap']['CTRL'][:,0,:],
                              activity['scap']['SCHZ'][:,1,:] - activity['scap']['SCHZ'][:,0,:]))

            pred = np.hstack((predicted_activity['scap']['CTRL'][:,1,:] - predicted_activity['scap']['CTRL'][:,0,:],
                              predicted_activity['scap']['SCHZ'][:,1,:] - predicted_activity['scap']['SCHZ'][:,0,:]))

            # select roi
            real = real[roi,:]
            pred = pred[roi,:]

            r[i] = pearsonr(real,pred)[0]
            MAE[i] = mean_absolute_error(real,pred)
            MAPE[i] = mean_absolute_perc_error(real,pred)
            Rsqr[i] = r2_score(real,pred)
    return r,MAE,MAPE,Rsqr