#!/bin/bash
#SBATCH --job-name=wtknn-sv
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

python applications_knn.py --task mislabel_detect --dataset $1 --value_type fastWTNN-SV --n_data 200 --n_val $2 --flip_ratio 0.1 --tau $3 --kernel $4 --n_repeat 1 >> wtnn_result/fWTNNSV-Mislabel-$1-Ntrain200-Nval$2-Tau$3-Kernel-$4-New.txt

