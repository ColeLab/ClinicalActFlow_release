# -*- coding: utf-8 -*-
"""
Simple script that checks if ciftify has been generated correctly and deletes fmriprep data that has not worked
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd

## INPUTS---
preprodir = '/projects/f_mc1689_1/ClinicalActFlow/data/prepro/'
ciftdir = preprodir + 'ciftify/'
fsdir = preprodir + 'freesurfer/'

## --------- Ciftify missing data

# change directory
origdir = os.getcwd()
os.chdir(ciftdir)

# get list of subjects in ciftify dir
subjects = glob.glob('sub*')
output = np.zeros((len(subjects),3))
for i in range(0,len(subjects)):
    s = subjects[i]

    #check folder structure is there
    file = s + '/MNINonLinear/Results/task-rest_bold/task-rest_bold_Atlas_s0.dtseries.nii'
    output[i,0] = os.path.isfile(file)
    
    file = s + '/T1w/'
    output[i,1] = os.path.isdir(file)
    
idx = np.sum(output,axis=1)<2
output[:,2] = idx

# put in dataframe
df = pd.DataFrame(data=output,index=subjects,columns=['MNInonlinear','T1w','Keep'])
df['subjects'] = subjects
dfprint = df[df['Keep'] == 1]
print dfprint.head()
print len(dfprint)

# delete the data associated with those subjects...
for i in range(0,len(dfprint)):
    s = dfprint['subjects'][i]
    print s
    print i
    print 'removing...'
    
    folder = ciftdir + s +'/'
    print folder
    #shutil.rmtree(folder)

## --------- freesurfer exists, but ciftify doesn't
# change directory
origdir = os.getcwd()
os.chdir(fsdir)

# get list of subjects in ciftify dir
subjects = glob.glob('sub*')
output = np.zeros((len(subjects),1))
for i in range(0,len(subjects)):
    s = subjects[i]

    #check if ciftify exists
    file = ciftdir + s
    output[i] = os.path.isdir(file)


# put in dataframe
df = pd.DataFrame(data=output,index=subjects,columns=['Ciftify'])
df['subjects'] = subjects
dfprint = df[df['Ciftify'] == 0]
print dfprint.head()


# delete the data associated with those subjects...
for i in range(0,len(dfprint)):
#for i in range(0,1):
    s = dfprint['subjects'][i]
    print s
    
    print 'removing...'
    
    folder = fsdir + s +'/'
    print folder
    #shutil.rmtree(folder)
    #os.rmdir(folder)
    
    folder = preprodir + '/fmriprep/' + s +'/'
    print folder
    #shutil.rmtree(folder)
    #os.rmdir(folder)