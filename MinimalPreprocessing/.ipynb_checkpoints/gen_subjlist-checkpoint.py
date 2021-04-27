# -*- coding: utf-8 -*-
"""
Simple script that takes a bids directory and a list of  task names and generates a subject list txt file
"""

import os
import glob
import numpy as np
import pandas as pd

## INPUTS---
bidsdir = '/projects/f_mc1689_1/ClinicalActFlow/data/ds000030_R105/'
#bidsdir = '/Users/luke/Documents/Projects/ClinicalActFlow/data/rawdata/'
ciftdir = '/projects/f_mc1689_1/ClinicalActFlow/data/prepro/ciftify/'
#ciftdir = '/Users/luke/Documents/Projects/ClinicalActFlow/data/prepro/ciftify/'

## ---------

# change directory
origdir = os.getcwd()
os.chdir(bidsdir)

# get list of subjects in bidsdir
subjects = glob.glob('sub*')
output = np.zeros((len(subjects),2))
for i in range(0,len(subjects)):
    s = subjects[i]

    #check func data
    file = s + '/func/' + s + '_task-rest_bold.json'
    output[i,0] = os.path.isfile(file)
    
    #check anat data
    file = s + '/anat/' + s + '_T1w.nii.gz'
    output[i,1] = os.path.isfile(file)

# remove subjects without rest data
df = pd.DataFrame(data=output,index=subjects,columns=['rest','anat'])
df['subjects'] = subjects
print(len(df),' subjects prior to RS exclusion')
df = df[df['rest'] != 0]
df = df[df['anat'] != 0]
df_subjects = df['subjects']
print(len(df),' subjects after RS exclusion')

# get list of subjects in output directory (ciftdir)
output = np.zeros((len(df_subjects)))
for i in range(0,len(df_subjects)):
    s = df_subjects[i]
    file = ciftdir + s + '/MNINonLinear/Results/task-rest_bold/task-rest_bold_Atlas_s0.dtseries.nii'
    output[i] = os.path.isfile(file)
    if output[i] == 1:
        print('subject ' + s + ' already has ciftify data - no need to redo')

df_subjects_postciftify = df_subjects.drop(df_subjects.index[np.where(output)])

#write the text file that will be used
os.chdir(origdir)
df_subjects_postciftify.to_csv('subject_list.txt', sep='\t',header = False, index = False)

#write the same subjects for ciftify (doesn't like the sub- prefix)
#df_subjectsNOSUB = df_subjects.str.split('-',expand=True)
#del df_subjectsNOSUB[0]
#df_subjects.to_csv('subject_list_cift.txt', sep='\t',header = False, index = False)