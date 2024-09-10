#!/bin/bash
#SBATCH --job-name=WKNNSV-Mislabel-Detection
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:59:59

#SBATCH --output=/scratch/gpfs/tw8948/slurm-%j.out
#SBATCH --error=/scratch/gpfs/tw8948/slurm-%j.out

if [ "$9" -eq 1 ]; then
    python applications_knn.py --task mislabel_detect --dataset $1 --value_type fastWKNN-SV --n_data $2 --n_val $3 --flip_ratio ${11} --K $4 --kernel $5 --dis_metric $6 --eps $7 --n_bits $8 --n_repeat 1 --temp ${10} --normalize --dataset_seed ${12} >> wtnn_result/WKNNSV-Mislabel-$1-Ntrain$2-Nval$3-K$4-Kernel-$5-dist-$6-eps$7-NB$8-temp${10}-FR${11}-Seed${12}-norm.txt
elif [ "$9" -eq 0 ]; then
    python applications_knn.py --task mislabel_detect --dataset $1 --value_type fastWKNN-SV --n_data $2 --n_val $3 --flip_ratio ${11} --K $4 --kernel $5 --dis_metric $6 --eps $7 --n_bits $8 --n_repeat 1 --temp ${10} --dataset_seed ${12} >> wtnn_result/WKNNSV-Mislabel-$1-Ntrain$2-Nval$3-K$4-Kernel-$5-dist-$6-eps$7-NB$8-temp${10}-FR${11}-Seed${12}.txt
else
    echo "The 9th argument must be 0 or 1."
    exit 1
fi
