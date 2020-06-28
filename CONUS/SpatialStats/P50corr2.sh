#!/bin/bash

#SBATCH --job-name=p50c2
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-140
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> P50corr2.out
python P50corr2.py
~