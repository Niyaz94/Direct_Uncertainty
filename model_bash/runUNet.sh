#!/bin/bash -l

#SBATCH -J Prepare

#SBATCH -N 1

#SBATCH --ntasks-per-node=10

#SBATCH --mem-per-cpu=8GB

#SBATCH --time=01:10:00 

##SBATCH -A plgaimed-gpu-a100

#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu

##SBATCH --output="output_pred1.out"

##SBATCH --error="error_pred1.err"


cd $SLURM_SUBMIT_DIR

srun /bin/hostname

module add monai/1.2.0
source /U-Net/unet-env/bin/activate

papermill  utility/METRICS.ipynb utility/METRICS_RUN.ipynb > log/result/output.log




