#!/bin/bash

#SBATCH --job-name=p50corr
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=0:40:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> P50corr.out
python P50corr.py
~
