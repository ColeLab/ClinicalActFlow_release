#!/usr/bin/env bash
# !/bin/bash

## AMAREL computing cluster (at Rutgers University) batch script
## template from Mike Cole adapted by Luke Hearne

#project directory
projDir='/projects/f_mc1689_1/ClinicalActFlow/'

#scriptDir
scriptDir=${projDir}docs/scripts/MinimalPreprocessing

#individual-level job scripts
subjBatchScriptDir=${scriptDir}/BatchScripts/

#Name you'll see when using squeue, sinfo, and sacct
jobNamePrefix="Cafpp-"

#list of subjects
listOfSubjects=$(<subject_list.txt)

#Make and execute a batch script for each subject
for subj in $listOfSubjects
do
    cd ${subjBatchScriptDir}

    batchFilename=${subj}_preproBatch.sh

    #Job specs (resources) indicated for each batch process
    echo "#!/bin/bash" > $batchFilename
    echo "#SBATCH --partition=p_mc1689_1" >> $batchFilename
    echo "#SBATCH --requeue" >> $batchFilename
    echo "#SBATCH --time=36:00:00" >> $batchFilename
    echo "#SBATCH --nodes=1" >> $batchFilename
    echo "#SBATCH --ntasks=1" >> $batchFilename
    echo "#SBATCH --job-name=${jobNamePrefix}${subj}" >> $batchFilename
    echo "#SBATCH --output=slurm.${jobNamePrefix}${subj}.out" >> $batchFilename
    echo "#SBATCH --error=slurm.${jobNamePrefix}${subj}.err" >> $batchFilename
    echo "#SBATCH --cpus-per-task=8" >> $batchFilename
    echo "#SBATCH --mem-per-cpu=12000" >> $batchFilename
    echo "#SBATCH --export=ALL" >>$batchFilename
    
    #batch script contents
    echo "module purge" >> $batchFilename
    echo "##Run the prepro script" >> $batchFilename
    echo "cd $scriptDir" >> $batchFilename
    
    # pass the subject name to the prepro script
    echo "./prepro_pipeline.sh '${subj}' " >> $batchFilename

    #Submit the job
    sbatch $batchFilename
done
