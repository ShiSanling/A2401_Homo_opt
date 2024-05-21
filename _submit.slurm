
#SBATCH -J cupy
#SBATCH -p debuga100
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH -o log/%j.out
#SBATCH -e log/%j.err

ulimit -s unlimited
ulimit -l unlimited
ulimit -c unlimited


python test1.py