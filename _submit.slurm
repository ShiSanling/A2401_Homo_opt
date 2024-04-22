#!/bin/bash

#SBATCH --job-name=python_Application
#SBATCH --partition=debug64c512g
#SBATCH --ntasks-per-node=64
#SBATCH -n 64
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

ulimit -s unlimited
ulimit -l unlimited
ulimit -c unlimited


mpiexec -n 64 ./ksp -Ain A.txt -rhs b.txt