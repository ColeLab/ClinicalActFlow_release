#!/bin/env bash

# we want to delete the working directory + any remaining freesurfer + fmriprep files in the case of a failed preprocessing
# failure is from being bumped from amarel

## inputs
subject=$1 #"e.g., sub-10159"

#freesurfer
f="/projects/f_mc1689_1/ClinicalActFlow/data/prepro/freesurfer/"${subject}
if [ -d ${f} ]; then
    echo "...removing "${f}
    rm -r ${f}
fi

#fmriprep
f="/projects/f_mc1689_1/ClinicalActFlow/data/prepro/fmriprep/"${subject}
if [ -d ${f} ]; then
    echo "...removing "${f}
    rm -r ${f}
fi

#ciftify
f="/projects/f_mc1689_1/ClinicalActFlow/data/prepro/ciftify/"${subject}
if [ -d ${f} ]; then
    echo "...removing "${f}
    rm -r ${f}
fi

#scratch
f="/projects/f_mc1689_1/ClinicalActFlow/scratch/reportlets/fmriprep/"${subject}
if [ -d ${f} ]; then
    echo "...removing "${f}
    rm -r ${f}
fi

#get subject name without 'sub-' prefix
oldIFS=$IFS
IFS='-'
read -ra ADDR <<<"$subject"
nsubject=${ADDR[1]}
IFS=${oldIFS}

f="/projects/f_mc1689_1/ClinicalActFlow/scratch/fmriprep_wf/single_subject_"${nsubject}"_wf"
if [ -d ${f} ]; then
    echo "...removing "${f}
    rm -r ${f}
fi