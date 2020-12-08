#!/bin/bash

#SBATCH --job-name=cvg
#SBATCH --output=JobInfo_FW/%x_%a.out
#SBATCH --error=JobInfo_FW/%x_%a.err
#SBATCH --array=0-94
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=1000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Convergence.out
python Convergence.py
~
