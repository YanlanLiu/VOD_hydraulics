#!/bin/bash

#SBATCH --job-name=fcls
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-87
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Forward_by_cluster.out
python Forward_by_cluster.py
~
