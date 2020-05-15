#!/bin/bash

#SBATCH --job-name=R510
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-999
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval.out
python Retrieval.py 7
python Retrieval.py 8
python Retrieval.py 9
python Retrieval.py 10
python Retrieval.py 11
python Retrieval.py 12
python Retrieval.py 13


