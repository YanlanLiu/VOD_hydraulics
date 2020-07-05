#!/bin/bash

#SBATCH --job-name=Ra705
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_new.out
python Retrieval_new.py 0
python Retrieval_new.py 1
python Retrieval_new.py 2
python Retrieval_new.py 3
python Retrieval_new.py 4
python Retrieval_new.py 5
python Retrieval_new.py 6

