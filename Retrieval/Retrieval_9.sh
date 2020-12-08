#!/bin/bash

#SBATCH --job-name=glb9
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_9.out
python Retrieval_0817.py 9
python Retrieval_0817.py 21
python Retrieval_0817.py 33
python Retrieval_0817.py 45
python Retrieval_0817.py 57
python Retrieval_0817.py 69
python Retrieval_0817.py 81
python Retrieval_0817.py 93
