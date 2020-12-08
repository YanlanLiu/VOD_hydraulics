#!/bin/bash

#SBATCH --job-name=glb4
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_4.out
python Retrieval_0817.py 4
python Retrieval_0817.py 16
python Retrieval_0817.py 28
python Retrieval_0817.py 40
python Retrieval_0817.py 52
python Retrieval_0817.py 64
python Retrieval_0817.py 76
python Retrieval_0817.py 88
