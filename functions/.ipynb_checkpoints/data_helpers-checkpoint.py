import numpy as np
import pandas as pd
import h5py
from IPython.display import clear_output

proj_dir = '/projects/f_mc1689_1/ClinicalActFlow/'

def load_activity(df,PARC='cabn',TASKS=['scap','taskswitch','bart','stopsignal'],pipeline='24pXaCompCorXVolterra',noncircular=False):
    
    data_dir = proj_dir + 'data/prepro/Output_'+PARC+'/'
    subj_list = df.index
    
    # load activity
    activity = {}
    activity_all = {}
    GLMmethod='canonical'
    
    for task in TASKS:
        print('Task|',task)
        # get size of contrasts & regions
        h5f = h5py.File(data_dir + subj_list[0] + '_GLMOutput.h5','r')
        if noncircular is False:
            data = h5f[task][GLMmethod]['taskActivity_betas_'+pipeline][:].copy()
            activity_task = np.zeros((np.shape(data)[0],np.shape(data)[1],len(subj_list)))
        elif noncircular is True:
            data = h5f[task][GLMmethod]['taskActivity_betas_'+pipeline+'-noncircular'][:].copy()
            activity_task = np.zeros((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2],len(subj_list)))
        h5f.close()
        
        #get data
        for s,subj in enumerate(subj_list):
            print(subj)
            h5f = h5py.File(data_dir + subj + '_GLMOutput.h5','r')
            if noncircular is False:
                data = h5f[task][GLMmethod]['taskActivity_betas_'+pipeline][:].copy()
                activity_task[:,:,s] = data
            elif noncircular is True:
                data = h5f[task][GLMmethod]['taskActivity_betas_'+pipeline+'-noncircular'][:].copy()
                activity_task[:,:,:,s] = data
            h5f.close()
            clear_output()
            print('Task|',task,'Data loaded:',np.round((s/len(subj_list))*100),'%')
            
        # save to array (no groups)
        activity_all[task] = activity_task

        if noncircular is False:
            activity[task] = {}
            activity[task]['CTRL'] = activity_task[:,:,df['group']=='CTRL']
            activity[task]['SCHZ'] = activity_task[:,:,df['group']=='SCHZ']
            activity[task]['BPLR'] = activity_task[:,:,df['group']=='BPLR']
            activity[task]['ADHD'] = activity_task[:,:,df['group']=='ADHD']
        elif noncircular is True:
            activity[task] = {}
            activity[task]['CTRL'] = activity_task[:,:,:,df['group']=='CTRL']
            activity[task]['SCHZ'] = activity_task[:,:,:,df['group']=='SCHZ']
            activity[task]['BPLR'] = activity_task[:,:,:,df['group']=='BPLR']
            activity[task]['ADHD'] = activity_task[:,:,:,df['group']=='ADHD']
    return activity,activity_all

def load_fc(df,TASKS=['rest'],PARC='cabn',fc_method='pearsoncorr',model_rest='24pXaCompCorXVolterra-spikeReg',model_task='24pXaCompCorXVolterra'):
    data_dir = proj_dir + 'data/prepro/Output_'+PARC+'/'
    subj_list = df.index
    
    fc = {}
    fc_all = {}
    for task in TASKS:
        if task=='rest':
            model = model_rest
        else:
            model = model_task
        # get size
        h5f = h5py.File(data_dir + subj_list[0] + '_FCOutput.h5','r')
        data = h5f[task]['fc'+model][fc_method][:].copy()
        h5f.close()
        fc_task = np.zeros((np.shape(data)[0],np.shape(data)[0],len(subj_list)))

        #get FC data
        for s,subj in enumerate(subj_list):
            print(subj)
            h5f = h5py.File(data_dir + subj + '_FCOutput.h5','r')
            data = h5f[task]['fc'+model][fc_method][:].copy()
            h5f.close()
            fc_task[:,:,s] = data.copy()
            clear_output()
            print('Task|',task,'Data loaded:',np.round((s/len(subj_list))*100),'%')
            
        fc_all[task] = fc_task
        
        #split into a list of groups
        fc[task] = {}
        fc[task]['CTRL'] = fc_task[:,:,df['group']=='CTRL']
        fc[task]['SCHZ'] = fc_task[:,:,df['group']=='SCHZ']
        fc[task]['BPLR'] = fc_task[:,:,df['group']=='BPLR']
        fc[task]['ADHD'] = fc_task[:,:,df['group']=='ADHD']
    return fc,fc_all

def load_fc_variance(subj_list,task='multitask-no-scap',PARC='cabn',fc_method='pc_multregconn_100',model='24pXaCompCorXVolterra'):
    
    data_dir = proj_dir + 'data/prepro/Output_'+PARC+'/'
    var = {}
    var_all = []

    #get variance
    for s,subj in enumerate(subj_list):
        h5f = h5py.File(data_dir + subj + '_FCOutput.h5','r')
        data = h5f[task]['fc'+model][fc_method+'_variance'][()].copy()
        h5f.close()
        var_all.append(data.copy())
    var_all = np.array(var_all)
    #split into a list of groups
    df = pd.read_csv(proj_dir + 'docs/scripts/subject_list.txt', sep='\t', index_col = 0, header = 0)
    
    var['CTRL'] = var_all[df['group']=='CTRL']
    var['SCHZ'] = var_all[df['group']=='SCHZ']
    var['BPLR'] = var_all[df['group']=='BPLR']
    var['ADHD'] = var_all[df['group']=='ADHD']
    return var,var_all

def load_scan_behaviour(subj_list,task_list,group_index):
    '''
    Loads in-scanner task behaviour. Currently only coded up for the 'SCAP' task.
    For in-depth explanation of the tasks and the conditions modelled in the GLM
    see the postFmriprepPipelines.py script which does the GLM.
    
    returns: 
    - accuracy, 
    - reaction_time, 
    - accuracy_all, 
    - reaction_time_all, 
    - labels
    '''
    BIDS_DIR = proj_dir + 'data/ds000030_R105/'
    accuracy_all = {}
    reaction_time_all = {}
    accuracy = {}
    reaction_time = {}
    labels = {}

    for task in task_list:
        if task=='scap':
            # 12 conditions - each cognitive load and each delay
            task_label = ['load1-delay1.5','load1-delay3','load1-delay4.5',
                     'load3-delay1.5','load3-delay3','load3-delay4.5',
                     'load5-delay1.5','load5-delay3','load5-delay4.5',
                     'load7-delay1.5','load7-delay3','load7-delay4.5']

            raw_acc = np.zeros((4,12,len(subj_list))) # trial x condition x subject
            raw_rt = raw_acc.copy()

            for i,subj in enumerate(subj_list):
                # get task information
                df_task = pd.read_csv(BIDS_DIR + subj + '/func/' + subj + '_task-' + task + '_events.tsv', sep='\t', header=0)

                for trial in range(1,13):
                    #accuracy
                    acc = df_task['ResponseAccuracy'].values[df_task['trial_type']==trial]=='CORRECT'
                    acc = acc.astype(int)
                    raw_acc[:,trial-1,i] = acc
                    #rt
                    rt = df_task['ReactionTime'].values[df_task['trial_type']==trial]
                    raw_rt[:,trial-1,i] = rt

        # put in a task list
        accuracy_all[task] = raw_acc
        reaction_time_all[task] = raw_rt
        labels[task] = task_label

        # put in a list with groups
        accuracy[task] = {}
        accuracy[task]['CTRL'] = raw_acc[:,:,group_index=='CTRL']
        accuracy[task]['SCHZ'] = raw_acc[:,:,group_index=='SCHZ']
        accuracy[task]['BPLR'] = raw_acc[:,:,group_index=='BPLR']
        accuracy[task]['ADHD'] = raw_acc[:,:,group_index=='ADHD']

        reaction_time[task] = {}
        reaction_time[task]['CTRL'] = raw_rt[:,:,group_index=='CTRL']
        reaction_time[task]['SCHZ'] = raw_rt[:,:,group_index=='SCHZ']
        reaction_time[task]['BPLR'] = raw_rt[:,:,group_index=='BPLR']
        reaction_time[task]['ADHD'] = raw_rt[:,:,group_index=='ADHD']
    return accuracy, reaction_time, accuracy_all, reaction_time_all, labels


def load_ind_diff_behaviour(subj_list, beh_tsv_dict):
    '''
    Loads the 54 behavioural variables for the clinical actflow project.
    See https://ars.els-cdn.com/content/image/1-s2.0-S0006322319314751-mmc1.pdf
    Measures are within Table S1 (all measures that do not have an asterisk!)
    
    beh_tsv_dict: imported from beh_tsv_helper.py
    subj_list: list of subjects to use in analysis
    
    Ex. beh_data,beh_labels = load_behaviour(SUBJ_LIST, beh_tsv.all_subjs)
    '''
    BEHAV_DIR = proj_dir + 'data/ds000030_R105/phenotype/'
    
    beh_data = []
    beh_labels = []
    # loop through tests
    for file in beh_tsv_dict.keys():
        df_phenotype = pd.read_csv(BEHAV_DIR+file,delimiter='\t')
        subjs_idx = df_phenotype['participant_id'].isin(subj_list)
        
        for i,col_label in enumerate(beh_tsv_dict[file][0]):
            beh_data.append(df_phenotype[col_label].values[subjs_idx])
            beh_labels.append(beh_tsv_dict[file][2][0]+': '+beh_tsv_dict[file][1][i])
           
    #turn into array
    beh_data = np.array(beh_data)

    return beh_data, beh_labels

def beh_tsv_helper():
    '''
    See https://ars.els-cdn.com/content/image/1-s2.0-S0006322319314751-mmc1.pdf

    Available Dictionaries: 
    all_tsv - all behavioural measures included in Table S1 (except Remember-Know Task)
    all_subjs - measures in Table S1 without asterisks, meaning all subjects have that data
    schz - measures for which all good SCHZ subjects have data

    Keys: tsv filenames
    [0]: list of tsv column labels
    [1]: corresponding clean measure labels
    [2]: test/task name
    '''

    all_tsv = {
     'phenotype_asrs.tsv': [['asrs_score'],['ADHD symptoms'],['Adult Self-Report']],
     'phenotype_acds_adult.tsv': [['adult_attention','adult_hyperactivity'],
                                  ['Inattention','Hyperactivity'],['ADHD Clinical Diagnosis']],
     'phenotype_hopkins.tsv': [['hopkins_anxiety','hopkins_depression','hopkins_obscomp',
                                'hopkins_intsensitivity','hopkins_somatization'],
                               ['Anxiety','Depression','Obsessive compulsiveness',
                                'Somatization','Interpersonal sensitivity'],['Hopkins Symptoms']], 
     'phenotype_bprs.tsv': [['bprs_positive','bprs_negative','bprs_mania','bprs_depanx'],
                            ['Positive symptoms','Negative symptoms','Mania/disorganization',
                             'Depression/anxiety'],['Brief Psychiatric Rating']],
     'phenotype_hamilton.tsv': [['hamd_17'],['Total (#1-17)'],['Hamilton Depression']],
     'phenotype_ymrs.tsv': [['ymrs_score'],['Total'],['Young Mania']],
     'phenotype_saps.tsv': [['factor_delusions','factor_hallucinations','factor_bizarrebehav',
                             'factor_posformalthought'],
                            ['Delusions','Hallucinations','Bizarre behavior',
                             'Positive formal thought disorder'],['Positive Symptoms']],
     'phenotype_sans.tsv': [['factor_alogia','factor_anhedonia','factor_attention',
                             'factor_avolition','factor_bluntaffect'],
                            ['Alogia','Anhedonia','Attention','Avolition','Blunt affect'],
                            ['Negative Symptoms']],
     'phenotype_chapper.tsv': [['chapper_total'],['Perceptual aberrations'],['Chapman Psychosis-Proneness']], 
     'phenotype_chapsoc.tsv': [['chapsoc_total'],['Social anhedonia'],['Chapman Psychosis-Proneness']], 
     'phenotype_chapphy.tsv': [['chapphy_total'],['Physical anhedonia'],['Chapman Psychosis-Proneness']], 
     'phenotype_chapinf.tsv': [['chapinf_total'],['Infrequency'],['Chapman Psychosis-Proneness']], 
     'phenotype_bipolar_ii.tsv': [['bipollarii_mood','bipollarii_daydreaming','bipollarii_energy',
                                   'bipollarii_anxiety'],
                                  ['Mood lability','Daydreaming','Energy/activity','Social anxiety'],
                                  ['Bipolar II Risk Traits']], 
     'phenotype_golden.tsv': [['golden_sumscore'],['Schizoid-type personality'],
                              ['Golden & Meehl MMPI Items']], 
     'phenotype_chaphyp.tsv': [['chaphypo_total'],['Hypomanic personality'],
                               ['Eckblad and Chapman Hypomanic']], 
     'phenotype_tci.tsv': [['reward_dependence','persistance','novelty','harmavoidance'],
                           ['Reward dependence','Persistence','Novelty seeking','Harm avoidance'],
                           ['Temperament and Character']], 
     'phenotype_barratt.tsv': [['bis_2attimp','bis_2motimp','bis_2npimp'],
                               ['Attentional impulsivity','Motor impulsivity','Nonplanning'],
                               ['Barratt Impulsiveness']], 
     'phenotype_dickman.tsv': [['func_total','dysfunc_total'],
                               ['Functional impulsivity','Dysfunctional impulsivity'],
                               ['Dickman Impulsivity']], 
     'phenotype_eysenck.tsv': [['scorei','scorev','scoree'],['Impulsiveness','Venturesomeness','Empathy'],
                               ['Eysenck Impulsivity']], 
     'phenotype_mpq.tsv': [['mpq_score'],['Control'],['Multidimensional Personality']], 
     'phenotype_discounting.tsv': [['ddt_small_k','ddt_medium_k','ddt_large_k','ddt_total_k'],
                                   ['Small rewards','Medium rewards','Large rewards','Total'],
                                   ['Kerby Delay Discounting Task']],
     'phenotype_bart.tsv': [['bart_meanblueadjustedpumps','bart_meanredadjustedpumps',
                             'bart_meanadjustedpumps','bart_ratiomeanredtoblueadjpumps'],
                            ['Low risk pumps','High risk pumps','Total pumps','High to low ratio'],
                            ['Balloon Analog Risk Task']],
     'phenotype_cvlt.tsv': [['cvlt_sdf','cvlt_sdc','cvlt_ldf','cvlt_ldc','cvlt_ldh'],
                            ['Short delay free recall','Short delay cued recall',
                             'Long delay free recall','Long delay cued recall','Long delay recognition'],
                            ['CA Verbal Learning']], 
     'phenotype_sr.tsv': [['sr_acc_enc','sr_reaction_enc','sr_acc_rec','sr_reaction_rec'],
                          ['Encoding accuracy','Encoding RT','Recall accuracy','Recall RT'],
                          ['Scene Recognition']],
     'phenotype_rk.tsv': [['rk_krt'],['Know mean RT'],['Remember-Know']],
     'phenotype_wms.tsv': [['ssp_totalraw','vr1ir_totalraw','vr2dr_totalraw','vr2r_totalraw',
                            'ds_ldsf','ds_ldsb','ds_ldss'],
                           ['Symbol span','Visual reproduction immediate recall',
                            'Visual reproduction delayed recall','Visual reproduction recognition',
                            'Digit span forward','Digit span backward','Digit span sequencing '],
                           ['Wechsler Memory']], 
     'phenotype_smnm.tsv': [['smnm_main_mn','smnm_main_mdrt','smnm_manip_mn','smnm_manip_mdrt'],
                            ['Maintenance mean accuracy','Maintenance median RT',
                             'Manipulation mean accuracy','Manipulation median RT'],
                            ['Spatial Maintenance and Manipulation']],
     'phenotype_vmnm.tsv': [['vmnm_main_mn','vmnm_main_mdrt','vmnm_manip_mn','vmnm_manip_mdrt'],
                            ['Maintenance mean accuracy','Maintenance median RT',
                             'Manipulation mean accuracy','Manipulation median RT'],
                            ['Verbal Maintainance and Manipulation']],
     'phenotype_scap.tsv': [['scap1_correct_sum','scap1_correctrt_mean','scap3_correct_sum',
                             'scap3_correctrt_mean','scap5_correct_sum','scap5_correctrt_mean',
                             'scap7_correct_sum','scap7_correctrt_mean','scap_max_capac'],
                            ['Load 1 acc','Load 1 RT','Load 3 acc','Load 3 RT','Load 5 acc','Load 5 RT',
                             'Load 7 acc','Load 7 RT','Max capacity'],['Spatial Capacity']],
     'phenotype_vcap.tsv': [['vcap3_correct_sum','vcap3_correctrt_mean','vcap5_correct_sum',
                             'vcap5_correctrt_mean','vcap7_correct_sum','vcap7_correctrt_mean',
                             'vcap9_correct_sum','vcap9_correctrt_mean','vcap_max_capac'],
                            ['Load 3 acc','Load 3 RT','Load 5 acc','Load 5 RT','Load 7 acc','Load 7 RT',
                             'Load 9 acc','Load 9 RT','Max capacity'],['Verbal Capacity']], 
     'phenotype_wais.tsv': [['mr_totalraw','lns_totalraw','voc_totalraw'],
                            ['Matrix reasoning','Letter/number sequencing','Vocabulary'],
                            ['Wechsler Adult Intelligence']], 
     'phenotype_stroop.tsv': [['scwt_conflict_acc_effect','scwt_conflict_rt_effect'],
                              ['Interference accuracy','Interference RT'],['Stroop']],
     'phenotype_colortrails.tsv': [['crt_index'],['Interference index'],['Color Trail']], 
     'phenotype_stopsignal.tsv': [['sst_ses_quant_rt'],['Quantile RT'],['Stop Signal']],
     'phenotype_taskswitch.tsv': [['ts_accuracy','ts_interference','ts_costshort','ts_costlong'],
                                  ['Accuracy','Interference','Switching cost','Residual switching cost'],
                                  ['Task Switching']], 
     'phenotype_ant.tsv': [['ant_conflict_rt_effect'],['Interference RT'],['Attention Network Task']], 
     'phenotype_cpt.tsv': [['cpt_hits','cpt_md_h','cpt_fa'],
                           ['Hit rate','Hits median RT','False alarm rate'],['Go/No Go']], 
     'phenotype_dkefs.tsv': [['etotal'],['English verbal fluency'],['Delis-Kaplan Executive Function']],
     'phenotype_dkefs_spanish.tsv': [['dkefss_stotal'],['Spanish verbal fluency'],
                                     ['Delis-Kaplan Executive Function']]}

    #'phenotype_.tsv': [[],[],[]]

    all_subjs = {}
    all_subjs_files = ['phenotype_asrs.tsv','phenotype_hopkins.tsv','phenotype_chapper.tsv','phenotype_chapsoc.tsv','phenotype_chapphy.tsv','phenotype_chapinf.tsv','phenotype_bipolar_ii.tsv','phenotype_golden.tsv','phenotype_chaphyp.tsv','phenotype_tci.tsv','phenotype_barratt.tsv','phenotype_dickman.tsv','phenotype_eysenck.tsv','phenotype_mpq.tsv','phenotype_cvlt.tsv','phenotype_wms.tsv','phenotype_wais.tsv','phenotype_colortrails.tsv','phenotype_taskswitch.tsv','phenotype_ant.tsv','phenotype_cpt.tsv','phenotype_dkefs.tsv']

    for file in all_subjs_files:
        all_subjs[file] = all_tsv[file]


    schz = {}
    schz_files = ['phenotype_asrs.tsv','phenotype_acds_adult.tsv','phenotype_hopkins.tsv','phenotype_bprs.tsv','phenotype_hamilton.tsv','phenotype_ymrs.tsv','phenotype_saps.tsv','phenotype_sans.tsv','phenotype_chapper.tsv','phenotype_chapsoc.tsv','phenotype_chapphy.tsv','phenotype_chapinf.tsv','phenotype_bipolar_ii.tsv','phenotype_golden.tsv','phenotype_chaphyp.tsv','phenotype_tci.tsv','phenotype_barratt.tsv','phenotype_dickman.tsv','phenotype_eysenck.tsv','phenotype_mpq.tsv','phenotype_cvlt.tsv','phenotype_wms.tsv','phenotype_scap.tsv','phenotype_vcap.tsv','phenotype_wais.tsv','phenotype_colortrails.tsv','phenotype_stopsignal.tsv','phenotype_taskswitch.tsv','phenotype_ant.tsv','phenotype_cpt.tsv','phenotype_dkefs.tsv']

    for file in schz_files:
        schz[file] = all_tsv[file]


    screening_files = ['phenotype_admin.tsv','phenotype_colorvision.tsv','phenotype_demographics.tsv','phenotype_handedness.tsv','phenotype_health.tsv','phenotype_language.tsv','phenotype_medication.tsv','phenotype_scid.tsv','phenotype_spanish_vocab.tsv','phenotype_tbi.tsv','phenotype_visualacuity.tsv']    
    return all_tsv,all_subjs,schz

from scipy.stats import norm
def load_scap_dprime(subj_list,group_index):
    '''
    Loads d-prime for the scap task
    
    returns: 
    - d-prime,  
    - labels
    '''
    phenotype_dir = '/projects/f_mc1689_1/ClinicalActFlow/data/ds000030_R105/phenotype/'
    # load in dataframes
    df = pd.read_csv(phenotype_dir+'phenotype_scap.tsv',delimiter='\t')
    idx = df['participant_id'].isin(subj_list)
    df=df[idx]


    # 12 conditions - each cognitive load and each delay
    labels = ['load1-delay1.5','load1-delay3','load1-delay4.5',
             'load3-delay1.5','load3-delay3','load3-delay4.5',
             'load5-delay1.5','load5-delay3','load5-delay4.5',
             'load7-delay1.5','load7-delay3','load7-delay4.5']
    
    tp = (df['scap1_tp_sum'].values
          + df['scap3_tp_sum'].values
          + df['scap5_tp_sum'].values
          + df['scap7_tp_sum'].values)/24
    fp = (df['scap1_fp_sum'].values
          + df['scap3_fp_sum'].values
          + df['scap5_fp_sum'].values
          + df['scap7_fp_sum'].values)/24
    
    # clean up any inf vals (1s or 0s)
    tp[tp==1] = 0.999
    tp[tp==0] = 0.001

    fp[fp==1] = 0.999
    fp[fp==0] = 0.001
    
    # get dprime
    d = norm.ppf(tp) - norm.ppf(fp)
    
    # put in a list with groups
    dprime = {}
    for group in ['CTRL','SCHZ','BPLR','ADHD']:
        dprime[group] = d[group_index==group]
    return dprime,tp,fp, labels