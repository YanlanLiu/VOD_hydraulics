#!/bin/bash

#SBATCH --job-name=glb10
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_10.out
python Retrieval_0817.py 10
python Retrieval_0817.py 22
python Retrieval_0817.py 34
python Retrieval_0817.py 46
python Retrieval_0817.py 58
python Retrieval_0817.py 70
python Retrieval_0817.py 82
