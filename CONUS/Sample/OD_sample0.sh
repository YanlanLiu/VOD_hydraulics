#!/bin/bash

#SBATCH --job-name=od_ve
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-269
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> OD_sample.out
python OD_sample.py 0
~
