#!/bin/bash

#SBATCH --job-name=fnd
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> ND_forward.out
python ND_forward.py 0
python ND_forward.py 1
~
