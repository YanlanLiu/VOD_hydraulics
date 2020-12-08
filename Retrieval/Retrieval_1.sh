#!/bin/bash

#SBATCH --job-name=glb1
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_1.out
python Retrieval_0817.py 1
python Retrieval_0817.py 13
python Retrieval_0817.py 25
python Retrieval_0817.py 37
python Retrieval_0817.py 49
python Retrieval_0817.py 61
python Retrieval_0817.py 73
python Retrieval_0817.py 85
