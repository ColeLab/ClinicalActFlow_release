#!/bin/bash

#SBATCH --partition=nm3
#SBATCH --requeue
#SBATCH --time=00:40:00
#SBATCH --array=1-2%100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=array
#SBATCH --output=batch/PCA%A_%a.out
#SBATCH --error=batch/PCA%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --export=ALL

# print array number
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# get subject list
mapfile -t subj_array < batch_subj_list.txt
current_sub=${subj_array[$SLURM_ARRAY_TASK_ID]}
echo "Current subject: " ${current_sub}

python PCA-reg-comp-calc.py ${current_sub}