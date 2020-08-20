#!/bin/bash

#SBATCH --job-name=glb8
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_8.out
python Retrieval_0817.py 8
python Retrieval_0817.py 20
python Retrieval_0817.py 32
python Retrieval_0817.py 44
python Retrieval_0817.py 56
python Retrieval_0817.py 68
python Retrieval_0817.py 80
python Retrieval_0817.py 92
