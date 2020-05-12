#!/bin/bash

#SBATCH --job-name=r2t
#SBATCH --output=JobInfo_fwd/%x_%a.out
#SBATCH --error=JobInfo_fwd/%x_%a.err
#SBATCH --array=0-14
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=0:15:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> CalR2_test.out
python CalR2_test.py
~
