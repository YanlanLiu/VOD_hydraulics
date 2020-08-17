#!/bin/bash

#SBATCH --job-name=fod
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-539
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> OD_forward.out
python OD_forward.py 2
python OD_forward.py 3
~
