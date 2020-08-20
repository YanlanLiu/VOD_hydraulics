#!/bin/bash

#SBATCH --job-name=glb0
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_0.out
python Retrieval_0817.py 0
python Retrieval_0817.py 12
python Retrieval_0817.py 24
python Retrieval_0817.py 36
python Retrieval_0817.py 48
python Retrieval_0817.py 60
python Retrieval_0817.py 72
python Retrieval_0817.py 84
