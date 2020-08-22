#!/bin/bash

#SBATCH --job-name=glb6
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_6.out
python Retrieval_0817.py 6
python Retrieval_0817.py 18
python Retrieval_0817.py 30
python Retrieval_0817.py 42
python Retrieval_0817.py 54
python Retrieval_0817.py 66
python Retrieval_0817.py 78
python Retrieval_0817.py 90

