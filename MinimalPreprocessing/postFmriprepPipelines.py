# POST FMRIPREP PIPELINES

'''
These are a collection of functions to perform nuisnance and task regressions on fmriprep (i.e., BIDS organised data).

Most of this was coded by Takuya Ito, but Luke Hearne adapated it for fmriprep specifically.

'''

# Takuya Ito
# 09/11/2018

# Post-processing nuisance regression using Ciric et al. 2017 inspired best-practices
## OVERVIEW
# There are two main parts to this script/set of functions
# 1. "step1_createNuisanceRegressors"
#   Generates a variety of nuisance regressors, such as motionSpikes, aCompCor regressors, etc. that are essential to a subset of Ciric-style models, with the addition of some new combinations (e.g., aCompCor + spikeReg + movement parameters)
#   This is actually the bulk of the script, and takes quite a while to compute, largely due to the fact that we need to load in 4D time series from the raw fMRI data (in order to compute regressors such as global signal)
# 2. "step2_nuisanceRegression"
#   This is the function that actually performs the nuisance regression, using regressors obtained from step1. There are a variety of models to choose from, including:
#       The best model from Ciric et al. (2017) (e.g., 36p + spikeReg)
#       What I call the "legacy Cole Lab models", which are the traditional 6 motion parameters, gsr, wm and ventricle time series and all their derivatives (e.g., 18p)
#       There is also 16pNoGSR, which is the above, but without gsr and its derivative.
#   Ultimately, read below for other combinations; what I would consider the best option that does NOT include GSR is the default, called "24pXaCompCorXVolterra" - read below for what it entails...

# IMPORTANT: In general, only functions step1, step2 and the parameters preceding that will need to be edited. There are many helper functions below, but in theory, they should not be edited.
# Currently, this script is defaulted to create the nuisance regressors in your current working directory (in a sub directory), and the glm output in your current working directory
# The default is set to use data from the HCP352 QC'd data set, so will need to be updated accordingly.
# For now, this only includes extensive nuisance regression. Any task regression will need to be performed independently after this.

## EXAMPLE USAGE:
# import nuisanceRegressionPipeline as nrp
# nrp.step1_createNuisanceRegressors(nproc=8)
# nrp.step2_nuisanceRegression(nproc=5, model='24pXaCompCorXVolterra',spikeReg=False,zscore=False)

import os,sys,h5py,warnings,shutil
import numpy as np
import pandas as pd
import nibabel as nib
import multiprocessing as mp
import statsmodels.api as sm
import scipy.stats as stats
from scipy import signal
from IPython.display import clear_output
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
sys.path.append('/home/ljh118/f_mc1689_1/AnalysisTools/')
import ActflowToolbox as aft
warnings.simplefilter('ignore', np.ComplexWarning)

## Define GLOBAL variables (variables accessible to all functions) - these are project specific.
OVERWRITE = 1
TASKS = ['rest','bart','scap','stopsignal','taskswitch']
PARC = 'cabn'
TR = 2
FIR_length = 9 #TRS 18 s as in Cole et al., (2019)
DATA_DIR = '/projects/f_mc1689_1/ClinicalActFlow/data/prepro/ciftify/' #'directory to imaging data'
FMRIPREP_DIR = '/projects/f_mc1689_1/ClinicalActFlow/data/prepro/fmriprep/' # 'derivatives dir'
BIDS_DIR = '/projects/f_mc1689_1/ClinicalActFlow/data/ds000030_R105/'
NUIS_REG_DIR = '/projects/f_mc1689_1/ClinicalActFlow/data/prepro/nuisanceRegressors/' 
OUTPUT_DIR = '/projects/f_mc1689_1/ClinicalActFlow/data/prepro/Output_'+PARC+'/'

#subject list
df = pd.read_csv('subject_list.txt', sep='\t', index_col = 0, header = 0)
SUBJ_LIST = df.index

# # remove the regressors directory if it exists, then create if it doesn't exist (trivial to redo in BIDS)
# try:
#     shutil.rmtree(nuis_reg_dir)
#     os.makedirs(nuis_reg_dir) # Create directory if it doesn't exist
# except:
#     os.makedirs(nuis_reg_dir)

#os.makedirs(OUTPUT_DIR)

def step1_createNuisanceRegressors(task,nproc=4):
    """
    Function to generate subject-wise nuisance parameters in parallel
    This function first defines a local function (a function within this function) to generate each subject's nuisance regressors
    Then we use the multiprocessing module to generate regressors for multiple subjects at a time

    **Note: Parameters in this function may need to be edited for project-specific purposes. Sections in which editing should NOT be done are noted
    """
    
    
    # Make below function global, so it is accessible to the parallel process (don't change this)
    global _createNuisanceRegressorsSubject
    def _createNuisanceRegressorsSubject(subj):
        ## Potentially will need to be edited, according to project

        # This is the path and filename for the output regressors
        #nuisance_reg_filename = nuis_reg_dir + subj + '_nuisanceRegressors.h5'

        # Spike regression threshold, using relative root-mean-square displacement (in mm)
        spikeReg = .25

        print('creating nuisance regressors for subject', subj, '|task:', task)

        # This is the TSV file which contains the BOLD confounds from fmriprep
        df_confounds = pd.read_csv(FMRIPREP_DIR + subj + '/func/' + subj + '_task-' + task + '_bold_confounds.tsv', sep='\t', header=0) 

        # For all 12 movement parameters (6 regressors + derivatives)
        # x, y, z + 3 rotational movements
        motionParams = df_confounds[['X','Y','Z','RotX','RotY','RotZ']].values

        # The derivatives of the above movements
        motionParams_deriv  = np.zeros(motionParams.shape)
        motionParams_deriv[1:] = motionParams[1:,:] - motionParams[:-1,:]

        ####
        # DO NOT CHANGE THIS SECTION, IT IS NECESSARY FOR THE SCRIPT
        h5f = h5py.File(NUIS_REG_DIR + subj + '_nuisanceRegressors.h5','a')
        try:
            h5f.create_dataset(task + '/motionParams',data=motionParams)
            h5f.create_dataset(task + '/motionParams_deriv',data=motionParams_deriv)
        except:
            del h5f[task + '/motionParams'], h5f[task + '/motionParams_deriv']
            h5f.create_dataset(task + '/motionParams',data=motionParams)
            h5f.create_dataset(task + '/motionParams_deriv',data=motionParams_deriv)
        h5f.close()
        # END OF DO NOT CHANGE
        ####
        
        # Obtain relative root-mean-square displacement -- this will differ across preprocessing pipelines
        # To compute: np.sqrt(x**2 + y**2 + z**2), where x, y, and z are motion displacement parameters
        # e.g., x = x[t] - x[t-1]; y = y[t] - y[t-1]; z = z[t] - z[t-1]

        displacement = np.zeros(np.shape(motionParams))
        for tr in range(1,len(motionParams)):
            for dim in range(0,3):
                displacement[tr,dim] = motionParams[tr,dim] - motionParams[tr-1,dim];

        relativeRMS = np.sqrt(displacement[:,0]**2 + displacement[:,1]**2+ displacement[:,2]**2)
        
        #get FD
        FD = df_confounds['FramewiseDisplacement'].values
        #getDVARS
        DVARS = df_confounds['stdDVARS'].values

        ## Write to h5py
        h5f = h5py.File(NUIS_REG_DIR + subj + '_nuisanceRegressors.h5','a')
        try:
            h5f.create_dataset(task + '/relativeRMS',data=relativeRMS)
            h5f.create_dataset(task + '/FD',data=FD)
            h5f.create_dataset(task + '/DVARS',data=DVARS)
        except:
            del h5f[task + '/relativeRMS'], h5f[task + '/FD'], h5f[task + '/DVARS']
            h5f.create_dataset(task + '/relativeRMS',data=relativeRMS)
            h5f.create_dataset(task + '/FD',data=FD)
            h5f.create_dataset(task + '/DVARS',data=DVARS)
        h5f.close()
        
        # Calculate motion spike regressors using helper functions defined below
        _createMotionSpikeRegressors(relativeRMS, subj, task, spikeReg=spikeReg)
            
        ## Extract physiological noise signals
        global_signal1d = df_confounds['GlobalSignal'].values
        wm_signal1d = df_confounds['WhiteMatter'].values
        ventricle_signal1d = df_confounds['CSF'].values

        ## Create derivative time series (with backward differentiation, consistent with 1d_tool.py -derivative option)
        global_signal1d_deriv = np.zeros(global_signal1d.shape)
        global_signal1d_deriv[1:] = global_signal1d[1:] - global_signal1d[:-1]
        wm_signal1d_deriv = np.zeros(wm_signal1d.shape)
        wm_signal1d_deriv[1:] = wm_signal1d[1:] - wm_signal1d[:-1]
        ventricle_signal1d_deriv = np.zeros(ventricle_signal1d.shape)
        ventricle_signal1d_deriv[1:] = ventricle_signal1d[1:] - ventricle_signal1d[:-1]

        ## Write to h5py
        h5f = h5py.File(NUIS_REG_DIR + subj + '_nuisanceRegressors.h5','a')
        try:
            h5f.create_dataset(task + '/global_signal',data=global_signal1d)
            h5f.create_dataset(task + '/global_signal_deriv',data=global_signal1d_deriv)
            h5f.create_dataset(task + '/wm_signal',data=wm_signal1d)
            h5f.create_dataset(task + '/wm_signal_deriv',data=wm_signal1d_deriv)
            h5f.create_dataset(task + '/ventricle_signal',data=ventricle_signal1d)
            h5f.create_dataset(task + '/ventricle_signal_deriv',data=ventricle_signal1d_deriv)
        except:
            del h5f[task + '/global_signal'], h5f[task + '/global_signal_deriv'], h5f[task + '/wm_signal'], 
            h5f[task + '/wm_signal_deriv'], h5f[task + '/ventricle_signal'], h5f[task + '/ventricle_signal_deriv']
            h5f.create_dataset(task + '/global_signal',data=global_signal1d)
            h5f.create_dataset(task + '/global_signal_deriv',data=global_signal1d_deriv)
            h5f.create_dataset(task + '/wm_signal',data=wm_signal1d)
            h5f.create_dataset(task + '/wm_signal_deriv',data=wm_signal1d_deriv)
            h5f.create_dataset(task + '/ventricle_signal',data=ventricle_signal1d)
            h5f.create_dataset(task + '/ventricle_signal_deriv',data=ventricle_signal1d_deriv)
        h5f.close()
        
        ## Obtain aCompCor regressors
        # !Critical difference - fmriprep combines compcor signals so as to not erronously use non ventricle signal.
        aCompCor = df_confounds[['aCompCor00','aCompCor01','aCompCor02','aCompCor03','aCompCor04',]].values
        aCompCor_deriv = np.zeros(aCompCor.shape)
        aCompCor_deriv[1:,:] = np.real(aCompCor[1:,:]) - np.real(aCompCor[:-1,:])
        
        ## Write to h5py
        h5f = h5py.File(NUIS_REG_DIR + subj + '_nuisanceRegressors.h5','a')
        try:
            h5f.create_dataset(task + '/aCompCor',data=aCompCor)
            h5f.create_dataset(task + '/aCompCor_deriv',data=aCompCor_deriv)
        except:
            del h5f[task + '/aCompCor'], h5f[task + '/aCompCor_deriv']
            h5f.create_dataset(task + '/aCompCor',data=aCompCor)
            h5f.create_dataset(task + '/aCompCor_deriv',data=aCompCor_deriv)
        h5f.close()
        
        ## Extract ICA AROMA regressors
        idx = df_confounds.columns.str.contains('AROMA*')
        AROMA = df_confounds.loc[:,idx].values

        ## Write to h5py
        h5f = h5py.File(NUIS_REG_DIR + subj + '_nuisanceRegressors.h5','a')
        try:
            h5f.create_dataset(task + '/AROMA',data=AROMA)
 
        except:
            del h5f[task + '/AROMA']
            h5f.create_dataset(task + '/AROMA',data=AROMA)
        h5f.close()

    # Construct parallel processes to run the local function in parallel (subject-wise parallelization)
    # Outputs will be found in "nuis_reg_dir" parameter
    pool = mp.Pool(processes=nproc)
    pool.map_async(_createNuisanceRegressorsSubject,SUBJ_LIST).get()
    pool.close()
    pool.join()

def step2_nuisanceRegression(task,nproc=6, model='24pXaCompCorXVolterra',spikeReg=False,zscore=False, aggAROMA=False,framesToSkip=0):
    """
    Function to perform nuisance regression on each run separately
    This uses parallel processing, but parallelization occurs within each subject
    Each subject runs regression on each region/voxel in parallel, thus iterating subjects and runs serially

    Input parameters:
        subj    : subject number as a string
        run     : task run
        outputdir: Directory for GLM output, as an h5 file (each run will be contained within each h5)
        model   : model choices for linear regression. Models include:
                    1. 24pXaCompCorXVolterra [default]
                        Variant from Ciric et al. 2017.
                        Includes (64 regressors total):
                            - Movement parameters (6 directions; x, y, z displacement, and 3 rotations) and their derivatives, and their quadratics (24 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives, and their quadratics (40 regressors)
                    2. 18p (the lab's legacy default)
                        Includes (18 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - Global signal and its derivative (2 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    3. 16pNoGSR (the legacy default, without GSR)
                        Includes (16 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    4. 12pXaCompCor (Typical motion regression, but using CompCor (noGSR))
                        Includes (32 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives (no quadratics; 20 regressors)
                    5. 36p (State-of-the-art, according to Ciric et al. 2017)
                        Includes (36 regressors total - same as legacy, but with quadratics):
                            - Movement parameters (6 directions) and their derivatives and quadratics (24 regressors)
                            - Global signal and its derivative and both quadratics (4 regressors)
                            - White matter signal and its derivative and both quadratics (4 regressors)
                            - Ventricles signal and its derivative (4 regressors)
        spikeReg : spike regression (Satterthwaite et al. 2013) [True/False]
                        Note, inclusion of this will add additional set of regressors, which is custom for each subject/run
        zscore   : Normalize data (across time) prior to fitting regression
        nproc = number of processes to use via multiprocessing
    """
    
    # Iterate through each subject
    for subj in SUBJ_LIST:
        
        # Iterate through each task
        print('Running regression on subject', subj, '| task', task)
        print('\tModel:', model, 'with spikeReg:', spikeReg, '| zscore:=', zscore, '| Aggressive AROMA:=', aggAROMA)
            
        ## Load in data to be preprocessed - This needs to be a space x time 2d array
        if PARC == 'cabn':
            inputfile = DATA_DIR+subj+'/MNINonLinear/Results/'+'task-'+task+'_bold/' + 'task-' + task + '_bold_Atlas_s0_'+ PARC +'.ptseries.nii.tsv'
            data = np.loadtxt(inputfile,delimiter='\t')
        elif PARC == 'glasser':
            inputfile = DATA_DIR+subj+'/MNINonLinear/Results/'+'task-'+task+'_bold/' + 'task-' + task + '_bold_Atlas_s0_'+ PARC +'2016.ptseries.nii.tsv'
            data = np.loadtxt(inputfile,delimiter='\t')
        elif PARC == 'vertex':
            inputfile = DATA_DIR+subj+'/MNINonLinear/Results/'+'task-'+task+'_bold/' + 'task-' + task + '_bold_Atlas_s0.dtseries.nii'
            data = nib.cifti2.load(inputfile).get_data().T
        #elif config.parcOut == 'power':
        #    inputfile = config.fmriprepdir+subj+'/func/'+subj+'_task-'+task+'_bold_space-MNI152NLin2009cAsym_preprocpower264.tsv'
        #    data = np.loadtxt(inputfile,delimiter='\t').T
            
        # Run nuisance regression for this subject's task, using a helper function defined below
        # Data will be output in 'outputdir', defined above
        nuisanceRegressors,new_model = _nuisanceRegression(subj, task, data, OUTPUT_DIR, model=model,spikeReg=spikeReg,zscore=zscore,aggAROMA=aggAROMA, nproc=nproc)
        

        # Do the regression
        betas,residual_ts = regression(data.T,nuisanceRegressors,constant=True)
        print('\t\t', np.shape(residual_ts)[0], ' TRs (in resid)')
        print('\t\t', np.shape(residual_ts)[1], ' rois (in resid)')

        # Save the results
        h5f = h5py.File(OUTPUT_DIR + subj + '_GLMOutput.h5','a')
        outname1 = task + '/nuisanceReg_resid_' + new_model
        outname2 = task + '/nuisanceReg_betas_' + new_model
        try:
            h5f.create_dataset(outname1,data=residual_ts)
            h5f.create_dataset(outname2,data=betas)
        except:
            del h5f[outname1], h5f[outname2]
            h5f.create_dataset(outname1,data=residual_ts)
            h5f.create_dataset(outname2,data=betas)
        h5f.close()
        clear_output()

def _nuisanceRegression(subj, task, inputdata, OUTPUT_DIR,  model='24pXaCompCorXVolterra', spikeReg=False, zscore=False, aggAROMA=False,framesToSkip=0,nproc=8):
    """
    This function runs nuisance regression on the Glasser Parcels (360) on a single subjects run
    Will only regress out noise parameters given the model choice (see below for model options)

    Input parameters:
        subj    : subject number as a string
        task    : task
        outputdir: Directory for GLM output, as an h5 file (each task will be contained within each h5)
        model   : model choices for linear regression. Models include:
                    1. 24pXaCompCorXVolterra [default]
                        Variant from Ciric et al. 2017.
                        Includes (64 regressors total):
                            - Movement parameters (6 directions; x, y, z displacement, and 3 rotations) and their derivatives, and their quadratics (24 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives, and their quadratics (40 regressors)
                                - In this case, due to fmriprep, just 5 in total.
                    2. 18p (the legacy default)
                        Includes (18 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - Global signal and its derivative (2 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    3. 16pNoGSR (the legacy default, without GSR)
                        Includes (16 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    4. 12pXaCompCor (Typical motion regression, but using CompCor (noGSR))
                        Includes (32 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives (no quadratics; 20 regressors)
                                     - In this case, due to fmriprep, just 5 in total.
                    5. 36p (State-of-the-art, according to Ciric et al. 2017)
                        Includes (36 regressors total - same as legacy, but with quadratics):
                            - Movement parameters (6 directions) and their derivatives and quadratics (24 regressors)
                            - Global signal and its derivative and both quadratics (4 regressors)
                            - White matter signal and its derivative and both quadratics (4 regressors)
                            - Ventricles signal and its derivative (4 regressors)
                    6. 2p (can be used in conjuction with AROMA)
                            - WM, CSF and GSR
                    7. 2pNoGSR
                            -WM and CSF only
        spikeReg : spike regression (Satterthwaite et al. 2013) [True/False]
                        Note, inclusion of this will add additional set of regressors, which is custom for each subject/run
        zscore   : Normalize data (across time) prior to fitting regression
       aggAROMA  : Adds AROMA confounds as additional regressors. Can be combined with other prepocessing strategies.
        nproc = number of processes to use via multiprocessing
    """

    data = inputdata

    tMask = np.ones((data.shape[1],))
    tMask[:framesToSkip] = 0

    # Skip frames
    data = data[:,framesToSkip:]

    # Demean each run
    data = signal.detrend(data,axis=1,type='constant')
    
    # Detrend each run
    data = signal.detrend(data,axis=1,type='linear')
    tMask = np.asarray(tMask,dtype=bool)

    if zscore:
        data = stats.zscore(data,axis=1)

    nROIs = data.shape[0]

    # Load nuisance regressors for this data
    h5f = h5py.File(NUIS_REG_DIR + subj + '_nuisanceRegressors.h5','r')
    if model == '2p':
        # Global signal
        global_signal = h5f[task]['global_signal'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[task]['wm_signal'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[task]['ventricle_signal'][:].copy()
        # Create nuisance regressors design matrix
        nuisanceRegressors = np.vstack((global_signal,wm_signal,ventricle_signal)).T # Need to vstack, since these are 1d arrays
        
    elif model == '2pNoGSR':
        # white matter signal + derivatives
        wm_signal = h5f[task]['wm_signal'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[task]['ventricle_signal'][:].copy()
        # Create nuisance regressors design matrix
        nuisanceRegressors = np.vstack((wm_signal,ventricle_signal)).T # Need to vstack, since these are 1d arrays

    elif model=='24pXaCompCorXVolterra':
        # Motion parameters + derivatives
        motion_parameters = h5f[task]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[task]['motionParams_deriv'][:].copy()
        # WM aCompCor + derivatives
        aCompCor = h5f[task]['aCompCor'][:].copy()
        aCompCor_deriv = h5f[task]['aCompCor_deriv'][:].copy()
        # Ventricles aCompCor + derivatives
        #aCompCor_ventricles = h5f[task]['aCompCor_ventricles'][:].copy()
        #aCompCor_ventricles_deriv = h5f[task]['aCompCor_ventricles_deriv'][:].copy()
        # Create nuisance regressors design matrix
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor, aCompCor_deriv))
        quadraticRegressors = nuisanceRegressors**2
        nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))

    elif model=='18p':
        # Motion parameters + derivatives
        motion_parameters = h5f[task]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[task]['motionParams_deriv'][:].copy()
        # Global signal + derivatives
        global_signal = h5f[task]['global_signal'][:].copy()
        global_signal_deriv = h5f[task]['global_signal_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[task]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[task]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[task]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[task]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))

    elif model=='16pNoGSR':
        # Motion parameters + derivatives
        motion_parameters = h5f[task]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[task]['motionParams_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[task]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[task]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[task]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[task]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))

    elif model=='12pXaCompCor':
        # Motion parameters + derivatives
        motion_parameters = h5f[task]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[task]['motionParams_deriv'][:].copy()
        # WM aCompCor + derivatives
        aCompCor = h5f[task]['aCompCor'][:].copy()
        aCompCor_deriv = h5f[task]['aCompCor_deriv'][:].copy()
        # Ventricles aCompCor + derivatives
        #aCompCor_ventricles = h5f[task]['aCompCor_ventricles'][:].copy()
        #aCompCor_ventricles_deriv = h5f[task]['aCompCor_ventricles_deriv'][:].copy()
        # Create nuisance regressors design matrix
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor, aCompCor_deriv))

    elif model=='36p':
        # Motion parameters + derivatives
        motion_parameters = h5f[task]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[task]['motionParams_deriv'][:].copy()
        # Global signal + derivatives
        global_signal = h5f[task]['global_signal'][:].copy()
        global_signal_deriv = h5f[task]['global_signal_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[task]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[task]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[task]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[task]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
        quadraticRegressors = nuisanceRegressors**2
        nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))

    if spikeReg:
        # Obtain motion spikes
        try:
            motion_spikes = h5f[task]['motionSpikes'][:].copy()
            nuisanceRegressors = np.hstack((nuisanceRegressors,motion_spikes))
        except:
            print('\t\t... no spike regression needed')
        # Update the model name - to keep track of different model types for output naming
        model = model + '-spikeReg'

    if zscore:
        model = model + '-zscore'

    if aggAROMA:
        ## "Aggressive" (i.e., regression) AROMA denoising
        # obtain AROMA data
        AROMA = h5f[task]['AROMA'][:].copy()
        # add to nuisance regressors
        nuisanceRegressors = np.hstack((nuisanceRegressors,AROMA))
        model = model + '-aggAROMA'
    h5f.close()
    
    # Skip first X frames of nuisanceRegressors, too
    nuisanceRegressors = nuisanceRegressors[framesToSkip:,:].copy() 
    return nuisanceRegressors,model

def _createMotionSpikeRegressors(relative_rms, subj, task,  spikeReg=.25):
    """
    relative_rms-  time x 1 array (for HCP data, can be obtained from the txt file 'Movement_RelativeRMS.txt'; otherwise see Van Dijk et al. (2011) Neuroimage for approximate calculation
    task         -   Indicate which task this is
    spikeReg    -   generate spike time regressors for motion spikes, using a default threshold of .25mm FD threshold
    """

    nTRs = relative_rms.shape[0]

    motionSpikes = np.where(relative_rms>spikeReg)[0]
    if len(motionSpikes)>0:
        spikeRegressorsArray = np.zeros((nTRs,len(motionSpikes)))

        for spike in range(len(motionSpikes)):
            spike_time = motionSpikes[spike]
            spikeRegressorsArray[spike_time,spike] = 1.0

        spikeRegressorsArray = np.asarray(spikeRegressorsArray,dtype=bool)

        # Create h5py output
        h5f = h5py.File(NUIS_REG_DIR + subj + '_nuisanceRegressors.h5','a')
        try:
            h5f.create_dataset(task + '/motionSpikes',data=spikeRegressorsArray)
        except:
            del h5f[task + '/motionSpikes']
            h5f.create_dataset(task + '/motionSpikes',data=spikeRegressorsArray)
        h5f.close()

def regression(data,regressors,alpha=0,constant=True):
    """
    Taku Ito
    2/21/2019
    Hand coded OLS regression using closed form equation: betas = (X'X + alpha*I)^(-1) X'y
    Set alpha = 0 for regular OLS.
    Set alpha > 0 for ridge penalty
    PARAMETERS:
        data = observation x feature matrix (e.g., time x regions)
        regressors = observation x feature matrix
        alpha = regularization term. 0 for regular multiple regression. >0 for ridge penalty
        constant = True/False - pad regressors with 1s?
    OUTPUT
        betas = coefficients X n target variables
        resid = observations X n target variables
    """
    # Add 'constant' regressor
    if constant:
        ones = np.ones((regressors.shape[0],1))
        regressors = np.hstack((ones,regressors))
    X = regressors.copy()

    # construct regularization term
    LAMBDA = np.identity(X.shape[1])*alpha

    # Least squares minimization
    try:
        C_ss_inv = np.linalg.pinv(np.dot(X.T,X) + LAMBDA)
    except np.linalg.LinAlgError as err:
        C_ss_inv = np.linalg.pinv(np.cov(X.T) + LAMBDA)

    betas = np.dot(C_ss_inv,np.dot(X.T,data))
    # Calculate residuals
    resid = data - (betas[0] + np.dot(X[:,1:],betas[1:]))

    return betas, resid


def step2_taskRegression(task,task_model='canonical', model='24pXaCompCorXVolterra',spikeReg=False,zscore=False, aggAROMA=False,framesToSkip=0):
    """
    Collects nuisance regressors for a specific prepro model and task regressors for a specific task
    See step2nuisanceRegression for information about the nuisance regression
    See: https://f1000research.com/articles/6-1262/v2 for contrasts that were used in the data paper
    """
    
    # Iterate through each subject
    for subj in SUBJ_LIST:
        
        # Iterate through each task
        print('Running TASK:',task_model ,'regression on subject', subj, '| task', task)
        print('\tModel:', model, 'with spikeReg:', spikeReg, '| zscore:=', zscore, '| Aggressive AROMA:=', aggAROMA)
            
        ## Load in data to be preprocessed - This needs to be a space x time 2d array
        if PARC == 'cabn':
            inputfile = DATA_DIR+subj+'/MNINonLinear/Results/'+'task-'+task+'_bold/' + 'task-' + task + '_bold_Atlas_s0_'+ PARC +'.ptseries.nii.tsv'
            data = np.loadtxt(inputfile,delimiter='\t')
        elif PARC == 'glasser':
            inputfile = DATA_DIR+subj+'/MNINonLinear/Results/'+'task-'+task+'_bold/' + 'task-' + task + '_bold_Atlas_s0_'+ PARC +'2016.ptseries.nii.tsv'
            data = np.loadtxt(inputfile,delimiter='\t')
        elif PARC == 'vertex':
            inputfile = DATA_DIR+subj+'/MNINonLinear/Results/'+'task-'+task+'_bold/' + 'task-' + task + '_bold_Atlas_s0.dtseries.nii'
            data = nib.cifti2.load(inputfile).get_data().T
        nROIs = data.shape[0]
        nTR  = data.shape[1]
        
        # get nuisance regressors for this subject's task, using a helper function defined below
        nuisanceRegressors,_ = _nuisanceRegression(subj, task, data, OUTPUT_DIR, model=model,spikeReg=spikeReg,zscore=zscore,aggAROMA=aggAROMA)
        
        # get task information
        df_task = pd.read_csv(BIDS_DIR + subj + '/func/' + subj + '_task-' + task + '_events.tsv', sep='\t', header=0)
        conditions = get_condition_info(df_task,task)
        
        # run the specific glm
        if task_model == 'canonical':
            # convolve with HRF and downsample into TR space
            taskRegressors = np.zeros((nTR,len(conditions.keys())))
            for i,label in enumerate(conditions.keys()):
                taskRegressors[:,i] = HRF_from_onsets(conditions[label]['onsets'][:],conditions[label]['durations'][:],nTR,TR)

        elif task_model == 'FIR':
            taskRegressors = np.zeros((nTR,1))
            for i,label in enumerate(conditions.keys()):
                # find first TR of each event
                taskOnsets = np.round(conditions[label]['onsets']/TR).astype(int)

                #create design matrix for FIR
                currentRegressors = np.zeros((nTR,FIR_length))
                for reg in range(FIR_length):
                    # current TR in relation to onset of task (incrementally increasing)
                    idx = taskOnsets+reg
                    idx = np.delete(idx,np.where(idx>=nTR)[0]) # delete any indices outside the range of TR
                    currentRegressors[idx,reg] = 1
                taskRegressors = np.hstack((taskRegressors,currentRegressors))
        
            # To prevent SVD does not converge error, make sure there are no columns with 0s
            zero_cols = np.where(np.sum(taskRegressors,axis=0)==0)[0]
            taskRegressors = np.delete(taskRegressors, zero_cols, axis=1)
           
        if  OVERWRITE == 0 and os.path.isfile(OUTPUT_DIR + subj + '_GLMOutput.h5') == 1:
            print('File already exists and overwrite is false: skipping...')
        else:
            # do the regression with all regressors
            allRegs = np.hstack((nuisanceRegressors,taskRegressors))
            betas,residual_ts = regression(data.T, allRegs, constant=True)
            ntaskRegressors = int(taskRegressors.shape[1])
            betas = betas[-ntaskRegressors:,:].T # Exclude nuisance regressors
            print('\t\t', np.shape(residual_ts)[0], ' TRs (in resid)')
            print('\t\t', np.shape(residual_ts)[1], ' rois (in resid)')
            
            # save betas and residual to Task Regression
            h5f = h5py.File(OUTPUT_DIR + subj + '_GLMOutput.h5','a')
            outname1 = task + '/' + task_model + '/taskActivity_resid_' + model
            outname2 = task + '/' + task_model + '/taskActivity_betas_' + model

            try:
                h5f.create_dataset(outname1,data=residual_ts)
                h5f.create_dataset(outname2,data=betas)
            except:
                del h5f[outname1], h5f[outname2]
                h5f.create_dataset(outname1,data=residual_ts)
                h5f.create_dataset(outname2,data=betas)
            h5f.close()
            
def HRF_from_onsets(onsets,durations,nTR,TR,upsample_rate=0.01,convolve=True):
    '''
    onsets = onsets in seconds
    onset_length = onset length in seconds
    nTR = the total number of images in the scan
    TR = the TR in seconds
    upsample_rate = the upsampled units in seconds

    returns a TR length convolved vector and a non-convolved vector
    '''
    # set up HRF
    HRF = spm_hrf(upsample_rate,1)

    # vector of onsets (1's and 0's)
    onset_vector_up = np.zeros((np.int(nTR / (upsample_rate / TR))))
    
    # use the middle of the TR for downsampling
    TR_mid = np.int((TR / upsample_rate)/2)

    #upsample onsets and durations
    onsets_up = onsets / upsample_rate
    durations_up = durations / upsample_rate

    for trial in range(len(onsets_up)):

        # onsets and offsets in upsample space
        on = np.round(onsets_up[trial])
        off = np.round(onsets_up[trial] + durations_up[trial])
        
        # add to onsets vector in TR space
        try:
            onset_vector_up[np.arange(on,off,1,dtype=int)] = 1
        except:
            print('\t\t trial',trial,' not added - check original data')

    
    if convolve is True:
        #convolve HRF with binary onsets
        conv_onset_up = np.convolve(onset_vector_up,HRF)

        # shorten to length of scan
        conv_onset_up = conv_onset_up[0:len(onset_vector_up)]

        # downsample back to TR space by slicing convolved timeseries
        
        output = conv_onset_up[TR_mid:len(conv_onset_up):np.int(TR / upsample_rate)]
    else:  
        
        onset_down = onset_vector_up[TR_mid:len(conv_onset_up):np.int(TR / upsample_rate)]
        output = onset_down.copy() 
    return output

# def HRF_from_onsets(onsets,durations,nTR,TR,convolve=True):

#     # convert onsets to TR space
#     onsets_TR = np.zeros((nTR))

#     for trial in range(len(onsets)):

#         # onsets and offsets in TR space (rounded)
#         on_TR = np.round(onsets[trial] / TR)
#         off_TR = np.round((onsets[trial] + durations[trial]) / TR)

#         # add to onsets vector in TR space
#         onsets_TR[np.arange(on_TR,off_TR,1,dtype=int)] = 1
    
#     # Convolve onsets in TR space based on SPM canonical HRF (likely period of task-induced activity)
#     if convolve is True:
#         spm_hrfTR = spm_hrf(TR,oversampling=1)
#         convolved_onsets = np.convolve(onsets_TR,spm_hrfTR)
        
#         # trim to TR
#         return convolved_onsets[0:nTR]
#     else:
#         return onsets_TR
    
# def HRF_from_onsets(onsets,onset_length,nTR,TR,precision=100,convolve=True):
#     '''
#     onsets = onsets in seconds
#     onset_length = onset length in seconds
#     nTR = the total number of images in the scan
#     TR = the TR in seconds
#     precision = the number of milliseconds precision
    
#     returns a TR length convolved vector and a non-convolved vector
#     '''
#     #convert into TR space
#     onsets = np.round(onsets*precision)
#     onset_length = np.round(onset_length*precision)
#     onset_off = np.round(onsets + onset_length)

#     #up sampled binary vector
#     onset_up = np.zeros((nTR*(TR*precision),1))
#     for i in range(len(onsets)):
#         onset_up[onsets[i].astype(int):onset_off[i].astype(int)] = 1
 
#     if convolve == True:
#         # create upsampled SPM HRF
#         #HRF = spm_hrf(TR,oversampling=float(precision))
#         HRF = spm_hrf(1,oversampling=float(precision))
        
#         #convolve HRF with binary onsets
#         conv_onset_up = np.convolve(onset_up[:,0],HRF)
        
#         # shorten to length of timeseries
#         conv_onset_up = conv_onset_up[:nTR*TR*precision]

#         #downsample by slicing convolved timeseries
#         #m = np.int((precision*TR)/2) # move to middle of TR rather than start.
#         #conv_onset_down = conv_onset_up[m:len(conv_onset_up):np.int(precision*TR)]
        
#         # downsample by averaging:
#         conv_onset_down = np.mean(np.reshape(conv_onset_up,(nTR,TR*precision)),axis=1)
#         output = conv_onset_down.copy()
#     else: # do not convolve
#         # downsample by averaging:
#         onset_down = np.mean(np.reshape(onset_up,(nTR,TR*precision)),axis=1)
#         output = onset_down.copy()
#     return output

def get_condition_info(df_task,task):
    '''
    this function generates a dict - 'conditions' that contains ONSET and DURATION information
    for each condition in a given task.
    e.g.,
    conditions['A']['onsets'] = np.array of onsets
    conditions['A']['durations'] = np.array of durations same length as onsets
    
    Every study will need to alter this function to fit the onsets / timings of a given task.
    '''
    conditions = {}
    
    # clean the task_df of any bad onsets
    df_task.dropna(subset=['onset'],inplace=True)
    
    if task=='bart':
        # Balloon analog risk task (bart).

        # Participants were allowed to pump a series of virtual balloons. Experimental balloons (green) resulted either 
        # in an explosion or in a successful pump (no explosion and 5 points). Control (white) balloons did not result 
        # in points nor exploded. Participants could choose not to pump but to cash out and start with a new balloon.

        #For the Balloon Analog Risk Task (BART), we included 9 task regressors: for each condition (accept, explode,
        #reject), we added a regressor with equal amplitude and durations of 1 second on each trial. Furthermore, we 
        #included the same regressors with the amplitude modulated by the number of trials before explosions (perceived 
        #as the probability of explosions). The modulator was mean centered to avoid estimation problems due to 
        #collinearity. For the conditions that require a response (accept, reject), a regres- sor was added with equal 
        #amplitude, and the duration equal to the reaction time. These regressors were orthogonalised with their 
        #fixed-duration counterpart to separate the fixed effect of the trial and the effect covarying with the reaction 
        #time. A regressor is added for the control condition.

        # 4 conditions, control, balloon-accept, balloon-explode and balloon-cashout
        labels = ['CONTROL','ACCEPT','EXPLODE','CASHOUT']
        onset_length = 1 # all onsets are the same length

        # create a list with condition-wise onset and duration information
        for label in labels:
            conditions[label] = {}
            conditions[label]['onsets'] = []
            conditions[label]['durations'] = []

        # control condition onsets and durations
        conditions['CONTROL']['onsets'] = df_task['onset'].values[df_task['trial_type']=='CONTROL']
        conditions['CONTROL']['durations'] = np.zeros((np.shape(conditions['CONTROL']['onsets'])))
        conditions['CONTROL']['durations'][...] = onset_length

        # do the baloon conditions 
        idx_i = df_task['trial_type']=='BALOON'
        for j,label in enumerate(labels[1::]):
            idx_j = df_task['action']==label
            idx = idx_i.values & idx_j.values
            conditions[label]['onsets'] = df_task['onset'][idx].values
            conditions[label]['durations'] = np.zeros((np.shape(conditions[label]['onsets'])))
            conditions[label]['durations'][...] = onset_length

    elif task=='scap':
        #Spatial working memory task. / Spatial Capacity task (scap)

        #Subjects were shown an array of 1, 3, 5 or 7 circles pseudorandomly positioned around a central fixation cross. 
        #After a delay, subjects were shown a green circle and were asked to indicate whether the circle was in the same 
        #position as one of the target circled. In addition to the memory load, the delay period was manipulated with 
        #delays of 1.5, 3 or 4.5s. Half the trials were true-positive and half were true negative.

        #In the Spatial Capacity Task (SCAP), 25 task regressors were included. For each cognitive load (1 - 3 - 5 - 7) 
        #and each delay (1.5 - 3 - 4.5) with a correct response, two regressors were added: a regressor with fixed durations 
        #of 5 seconds and one with the duration equal to the reaction time, with the second orthogonalised with respect to 
        #the first. For both regressors, the onset is after the delay. The last regressor summarises all incorrect trials.

        # 12 conditions - each cognitive load and each delay with a 5 second regressor after the delay
        labels = list(range(1,13))

        for label in labels:
            conditions[label] = {}
            conditions[label]['onsets'] = []
            conditions[label]['durations'] = []
            
        for label in range(1,13):
            conditions[label]['onsets'] = df_task['onset'].values[df_task['trial_type']==label]
            conditions[label]['durations'] = df_task['duration'].values[df_task['trial_type']==label]
        
#         df = df_task.loc[df_task['ResponseAccuracy']=='CORRECT']
            
#         label = 'loWM'
#         conditions[label] = {}
#         conditions[label]['onsets'] = []
#         conditions[label]['durations'] = []

#         df_new = df.loc[(df['Load']==1) | (df['Load']==3)]
#         conditions[label]['onsets'] = df_new['onset'].values
#         conditions[label]['durations'] = df_new['duration'].values

#         label = 'hiWM'
#         conditions[label] = {}
#         conditions[label]['onsets'] = []
#         conditions[label]['durations'] = []
        
#         df_new = df.loc[(df['Load']==5) | (df['Load']==7)]
#         conditions[label]['onsets'] = df_new['onset'].values
#         conditions[label]['durations'] = df_new['duration'].values
        
#         label = 'incorrect'
#         conditions[label] = {}
#         conditions[label]['onsets'] = []
#         conditions[label]['durations'] = []
        
#         df_new = df_task.loc[df_task['ResponseAccuracy']=='INCORRECT']
#         conditions[label]['onsets'] = df_new['onset'].values
#         conditions[label]['durations'] = df_new['duration'].values

    elif task=='stopsignal':
        # Stop signal task. 
        # Participants were instructed to respond quickly when a ‘go’ stimulus was presented on the computer screen, except 
        # on the subset of trials where the ‘go’ stimulus was paired with a ‘stop’ signal. The ‘go’ stimulus was a pointing
        # arrow, a stop-signal was a 500 Hz tone presented through headphones.

        #For the Stop-Signal Task (STOPSIGNAL), for each condition (go, stop - successful, stop - unsuccessful), one task 
        #regressor was included with a fixed duration of 1.5s. For the conditions requir- ing a response (go and 
        #stop-unsuccessful), an extra regressor was added with equal amplitude, but the duration equal to the reaction time. 
        #Again, these regressors were orthogonalised with respect to the fixed duration regressor of the same condition. 
        #A sixth regressor was added with erroneous trials.

        # 2 conditions - stop - successful, stop - unsuccessful
        labels = ['SuccessfulStop','UnsuccessfulStop']
        onset_length = 1.5
        for label in labels:
            conditions[label] = {}
            conditions[label]['onsets'] = []
            conditions[label]['durations'] = []

        for label in labels:
            idx = df_task['TrialOutcome']==label
            conditions[label]['onsets'] = df_task['onset'][idx].values
            conditions[label]['durations'] = np.zeros((np.shape(conditions[label]['onsets'])))
            conditions[label]['durations'][...] = onset_length

    elif task=='taskswitch':
        # Task-switching task. 
        # Stimuli were shown varying in color (red or green) and in shape (triangle or shape). Participants were asked to 
        # respond to the stimulus based on the task cue (shape ‘S’ or color ‘C’). The task switched on 33% of the trials.
        # 09/05 Changed onset to Probeonset after reading original behavioural paper hidden in the poldrack paper...

        #In the Task Switching Task (TASKSWITCH), all manipulations were crossed (switch/no switch, congruent/incongruent, 
        #CSI delay short/long), resulting in 8 task conditions. As in the SCAP task, we added for each condition two 
        #regressors: a regressor with fixed durations of 1 second, and one with the duration equal to the reaction time, 
        #with the second orthogonalised with respect to the first. There is a total of 16 regressors.
        labels = list(range(0,8))
        onset_length = 1
        for label in labels:
            conditions[label] = {}
            conditions[label]['onsets'] = []
            conditions[label]['durations'] = []

        label=0
        for i in ['NOSWITCH','SWITCH']:
            idx_i = df_task['Switching']==i
            for j in ['CONGRUENT','INCONGRUENT']:
                idx_j = df_task['Congruency']==j
                for k in ['SHORT','LONG']:
                    idx_k = df_task['CSI']==k
                    idx = idx_i.values & idx_j.values & idx_k.values
                    conditions[label]['onsets'] = df_task['ProbeOnset'][idx].values
                    conditions[label]['durations'] = np.zeros((np.shape(conditions[label]['onsets'])))
                    conditions[label]['durations'][...] = onset_length
                    label+=1
    return conditions

def step3_calcFC(task,fc_method='pearsoncorr',model='24pXaCompCorXVolterra'):
    '''
    fcmethod = a string indicating what connectivity method to use. Options: 'multreg','pearsoncorr', 'pc_multregconn'

    
    '''
    for subj in SUBJ_LIST:
        print('Running fc for subject', subj, '| task', task)
        fc_method_label = fc_method
        h5f = h5py.File(OUTPUT_DIR + subj + '_GLMOutput.h5','r')

        if task == 'rest':
            # load residuals from specific parcellation
            data = h5f[task]['nuisanceReg_resid_'+model][:].copy()
        else:
            # get FIR data residuals
            data = h5f[task]['FIR']['taskActivity_resid_'+model][:].copy()
        h5f.close()
        
        # calculate FC
        if fc_method is 'pearsoncorr':
            fc = aft.connectivity_estimation.corrcoefconn(data.T)
        elif fc_method is 'multreg':
            fc = aft.connectivity_estimation.multregconn(data.T)
        elif fc_method is 'pc_multregconn':
            fc = aft.connectivity_estimation.pc_multregconn(data.T,n_components=np.min(np.shape(data))-1)
        elif fc_method is 'partial_corr':
            fc = aft.connectivity_estimation.partial_corrconn(data.T,estimator='LedoitWolf')
        else:
            print('Connectivity method unknown')
            
        # save data
        h5f = h5py.File(OUTPUT_DIR + subj + '_FCOutput.h5','a')
        outname1 = task + '/fc' + model + '/'+ fc_method_label

        try:
            h5f.create_dataset(outname1,data=fc)
        except:
            del h5f[outname1]
            h5f.create_dataset(outname1,data=fc)
        h5f.close()
        
# def step3_calcFCmulti(left_out_task=None,fc_method='pearsoncorr',model_rest='24pXaCompCorXVolterra-spikeReg',model='24pXaCompCorXVolterra',zscore=False):
#     '''
#     fcmethod = a string indicating what connectivity method to use. Options: 'multreg','pearsoncorr', 'pc_multregconn'

    
#     '''
#     for subj in SUBJ_LIST:
#         print('Running fc for subject', subj)
#         task_list = ['rest','scap','bart','stopsignal','taskswitch']
#         task_label = 'multitask'
#         fc_method_label = fc_method

#         if left_out_task is not None:
#             print('Leaving out',left_out_task)
#             # also delete from task list and change the task label.
#         if zscore:
#             task_label = task_label+'-zscore'
#         h5f = h5py.File(OUTPUT_DIR + subj + '_GLMOutput.h5','r')
#         data = np.zeros((718,1))
#         for task in task_list:
#             if task == 'rest':
#                 # load residuals from specific parcellation
#                 current_data = h5f[task]['nuisanceReg_resid_'+model_rest][:].copy()
#             else:
#                 # get FIR data residuals
#                 #print(h5f[task].keys())
#                 current_data = h5f[task]['FIR']['taskActivity_resid_'+model][:].copy()
                
#             # demean data
#             current_data = signal.detrend(current_data,axis=0,type='constant')
            
#             # zscore data
#             if zscore:
#                 current_data = stats.zscore(current_data,axis=0)
                
#             # concatenate data
#             data = np.hstack((data,current_data.T))
#         h5f.close()
#         # remove the initial zero col
#         data = np.delete(data,0,axis=1)

#         # calculate FC
#         if fc_method is 'pearsoncorr':
#             fc = aft.connectivity_estimation.corrcoefconn(data)
#         elif fc_method is 'multreg':
#             fc = aft.connectivity_estimation.multregconn(data)
#         elif fc_method is 'pc_multregconn':
#             fc = aft.connectivity_estimation.pc_multregconn(data)#,n_components=np.min(np.shape(data))-1)

#         # save data
#         h5f = h5py.File(OUTPUT_DIR + subj + '_FCOutput.h5','a')
#         outname1 = task_label + '/fc' + model + '/'+ fc_method_label

#         try:
#             h5f.create_dataset(outname1,data=fc)
#         except:
#             del h5f[outname1]
#             h5f.create_dataset(outname1,data=fc)
#         h5f.close()