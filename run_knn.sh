#!/bin/bash
#SBATCH --job-name=KNNSV
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:59:59

#SBATCH --constraint=skylake

#SBATCH --output=/scratch/gpfs/tw8948/slurm-%j.out
#SBATCH --error=/scratch/gpfs/tw8948/slurm-%j.out


python applications_knn.py --task mislabel_detect --dataset MNIST --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 5 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-MNIST-Ntrain50000-Nval1000-K5-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_CLIP --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 5 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_CLIP-Ntrain50000-Nval1000-K5-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_ImageNet --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 5 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_ImageNet-Ntrain50000-Nval1000-K5-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_CIFAR10 --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 5 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_CIFAR10-Ntrain50000-Nval1000-K5-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset MNIST --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 15 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-MNIST-Ntrain50000-Nval1000-K15-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_CLIP --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 15 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_CLIP-Ntrain50000-Nval1000-K15-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_ImageNet --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 15 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_ImageNet-Ntrain50000-Nval1000-K15-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_CIFAR10 --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 15 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_CIFAR10-Ntrain50000-Nval1000-K15-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset MNIST --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 25 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-MNIST-Ntrain50000-Nval1000-K25-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_CLIP --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 25 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_CLIP-Ntrain50000-Nval1000-K25-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_ImageNet --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 25 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_ImageNet-Ntrain50000-Nval1000-K25-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_CIFAR10 --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 25 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_CIFAR10-Ntrain50000-Nval1000-K25-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset MNIST --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 50 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-MNIST-Ntrain50000-Nval1000-K50-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_CLIP --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 50 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_CLIP-Ntrain50000-Nval1000-K50-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_ImageNet --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 50 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_ImageNet-Ntrain50000-Nval1000-K50-Kernel-plain-dist-l2.txt
python applications_knn.py --task mislabel_detect --dataset CIFAR10_CIFAR10 --value_type KNN-SV-RJ --n_data 50000 --n_val 1000 --flip_ratio 0.1 --K 50 --n_repeat 1 --dis_metric l2 >> wtnn_result/KNNSV-Mislabel-CIFAR10_CIFAR10-Ntrain50000-Nval1000-K50-Kernel-plain-dist-l2.txt

