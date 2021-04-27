#!/bin/env bash

## inputs
subject=$1 #"e.g., sub-10159"
./cleandirs.sh ${subject}

## containers
fmriprep_container=$HOME/containers/fmriprep-1.1.8.simg
ciftify_container="/home/ljh118/containers/ciftify_1.1.8-2.1.1.simg"

## paths
projectdir="/projects/f_mc1689_1/ClinicalActFlow/"
singdir="/mnt"

workdir=${singdir}/scratch
bidsdir=${singdir}/data/ds000030_R105
outdir=${singdir}/data/prepro

#path to freesurfer license
export FS_LICENSE=$HOME/license.txt

### FMRIPREP
# load singularity module
module purge
module load singularity/.2.5.1.99

# bind the data directory to singularity and call fmriprep
unset PYTHONPATH
singularity run --bind ${projectdir}:${singdir} \
${fmriprep_container} \
--participant-label ${subject} \
--output-space {T1w,template} \
--write-graph \
--use-aroma \
--use-syn-sdc \
--low-mem \
-n-cpus 8 \
--omp-nthreads 8 \
--mem-mb 8000 \
-w ${workdir} \
${bidsdir} \
${outdir} \
participant

### CIFTIFY
module purge
module load singularity/.2.5.1.99

#get subject name without 'sub-' prefix
oldIFS=$IFS
IFS='-'
read -ra ADDR <<<"$subject"
nsubject=${ADDR[1]}
IFS=${oldIFS}

singularity run --cleanenv \
-B ${projectdir}:/mnt \
-H ${HOME}:/tmp \
${ciftify_container} ${bidsdir} ${outdir} participant \
  	--participant_label=${nsubject} \
  	--fmriprep-workdir ${workdir} \
  	--n_cpus 8 \
  	--fs-license /tmp/license.txt \
