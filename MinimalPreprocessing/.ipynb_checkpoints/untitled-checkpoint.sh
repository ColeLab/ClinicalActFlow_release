#!/usr/bin/env bash

python gen_subjlist.py

value=$(<subject_list.txt)
echo "$value"
for subj in $value
do
    echo ${subj}
done


# manual
#list='sub-10171 sub-10189 sub-10206 sub-10217'
#echo ${list}