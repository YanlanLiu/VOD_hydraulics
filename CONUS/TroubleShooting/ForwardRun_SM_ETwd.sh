#!/bin/bash

#SBATCH --job-name=fsm_et
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=0:45:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> ForwardRun_SM_ETwd.out
python ForwardRun_SM_ETwd.py
~
