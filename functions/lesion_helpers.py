import numpy as np
from .act_flow_helpers import *

def network_lesion(activity,fc,held_out_roi,network_def,networks,roi_list=None,lesion_type='hold-out',task='scap',fc_task='multitask-no-scap'):
    '''
    activity in the shape used in clinicalActFlow ->activity[task][group][nodes,conditions,subj]
    network_def = vector of network affiliations
    networks = labels for networks (related to network_defs)
    '''
    #pre allocate
    predicted_activity_lesion = {}
    predicted_activity_lesion[task] = {}

    for i,net in enumerate(np.unique(network_def)):
        predicted_activity_lesion[task][networks[i]] = {}
        
        # set the lesion type
        if lesion_type == 'hold-in':
            # hold in a single network at a time for prediction
            net_index = network_def!=net
        elif lesion_type == 'hold-out':
            # hold out a single network at a time for prediction
            net_index = network_def==net
        
        for group in ['CTRL','SCHZ']:
            actPredVector = np.zeros(np.shape(activity[task][group]))

            n_nodes =  np.shape(actPredVector)[0]
            n_subs = np.shape(actPredVector)[2]
            n_conditions = np.shape(activity[task][group])[1]
            fc_data = fc[fc_task][group]

            for condition in range(n_conditions):
                act_data = activity[task][group][:,condition,:].copy()
                # delete the network related rois
                act_data[net_index,:] = 0

                for subj in range(np.shape(fc_data)[2]):
                    actPredVector[:,condition,subj],_ = actflowcalc_hold_out_roi(act_data[:,subj],fc_data[:,:,subj],held_out_roi=held_out_roi,roi_list=roi_list)

            predicted_activity_lesion[task][networks[i]][group] = actPredVector
    return predicted_activity_lesion