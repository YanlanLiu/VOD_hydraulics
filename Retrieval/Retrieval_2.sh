#!/bin/bash

#SBATCH --job-name=glb2
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_2.out
python Retrieval_0817.py 2
python Retrieval_0817.py 14
python Retrieval_0817.py 26
python Retrieval_0817.py 38
python Retrieval_0817.py 50
python Retrieval_0817.py 62
python Retrieval_0817.py 74
python Retrieval_0817.py 86
