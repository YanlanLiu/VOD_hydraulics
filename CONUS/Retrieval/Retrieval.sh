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
python Retrieval.py 0
python Retrieval.py 1
python Retrieval.py 2
python Retrieval.py 3
python Retrieval.py 4
python Retrieval.py 5
python Retrieval.py 6


