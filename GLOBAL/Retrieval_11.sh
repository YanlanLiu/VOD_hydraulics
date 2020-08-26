#!/bin/bash

#SBATCH --job-name=glb11
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-999
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_11.out
python Retrieval_0817.py 11
python Retrieval_0817.py 23
python Retrieval_0817.py 35
python Retrieval_0817.py 47
python Retrieval_0817.py 59
python Retrieval_0817.py 71
python Retrieval_0817.py 83
