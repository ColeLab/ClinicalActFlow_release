#!/bin/bash
module purge
module load singularity/.2.5.1.99

##inputs
subject=$1

#export freesufer_license=$HOME/license.txt  #place in home directory
#export FS_LICENSE='/home/ljh118/license.txt'
ciftify_container='/home/ljh118/containers/tigrlab_fmriprep_ciftify_1.1.8-2.1.1.simg'

## build the mounts
projectdir="/projects/f_mc1689_1/ClinicalActFlow/"
singdir='/mnt/'
bidsdir=${singdir}data/ds000030_R105
outdir=${singdir}data/prepro
workdir=${singdir}scratch

singularity run --cleanenv \
-B ${projectdir}:/mnt \
-H ${HOME}:/tmp \
${ciftify_container} ${bidsdir} ${outdir} participant \
  	--participant_label=${subject} \
  	--fmriprep-workdir ${workdir} \
  	--n_cpus 8 \
  	--fs-license /tmp/license.txt \
	--v

