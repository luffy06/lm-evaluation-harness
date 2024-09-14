#!/bin/bash
#SBATCH --job-name=attn-sparsity
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Shangyu.Wu@mbzuai.ac.ae
#SBATCH --mem=230G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --cpus-per-task=64 # Number of CPU cores
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=12:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs

date; hostname; pwd

bash attention_sparsity.sh