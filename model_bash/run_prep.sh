#!/bin/bash -l

#SBATCH -J MyJobNZJL

#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=48:00:00 
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu


#SBATCH --output="/U-Net/log/output/out_%j.out"
#SBATCH --error="/U-Net/log/error/err_%j.err"

## transition to the directory from which sbatch was called
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

module add monai/1.2.0

source /U-Net/unet-env/bin/activate

SCRIPT_PATH="/U-Net"

# DATASET_VALUES=("BRAIN_TUMOR" "KIDNEY" "KNEE" "PANCREATIC_LESION" "PROSTATE" "BRAIN_GROWTH" "PANCREAS" "SIJ" "LUNG" "HEART")
MULTIPLE_VALUES=("SINGLE" "MULTIPLE" "UNCERTAINTY_CLASS")

for DATASET in "${DATASET_VALUES[@]}"; do
    for MULTIPLE in "${MULTIPLE_VALUES[@]}"; do
        CONFIG_PATH="$SCRIPT_PATH/data/$DATASET/$MULTIPLE/configs/config_prep.json"
        echo "Running preprocessing for: $CONFIG_PATH"
        python3 $SCRIPT_PATH/run_preprocessing.py --config "$CONFIG_PATH"
    done
done