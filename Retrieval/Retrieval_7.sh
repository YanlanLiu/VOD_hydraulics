#!/bin/bash

#SBATCH --job-name=glb7
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_7.out
python Retrieval_0817.py 7
python Retrieval_0817.py 19
python Retrieval_0817.py 31
python Retrieval_0817.py 43
python Retrieval_0817.py 55
python Retrieval_0817.py 67
python Retrieval_0817.py 79
python Retrieval_0817.py 91
