#!/bin/bash -l

source /U-Net/unet-env/bin/activate

SCRIPT_PATH="/U-Net"


DATASET_VALUES=("BRAIN_TUMOR" "KIDNEY" "KNEE" "PANCREATIC_LESION" "PROSTATE" "BRAIN_GROWTH" "PANCREAS" "SIJ" "LUNG" "HEART")
MULTIPLE_VALUES=("SINGLE" "MULTIPLE" "UNCERTAINTY_CLASS")

for DATASET in "${DATASET_VALUES[@]}"; do
    for MULTIPLE in "${MULTIPLE_VALUES[@]}"; do
        CONFIG_PATH="$SCRIPT_PATH/data/$DATASET/$MULTIPLE/configs/config_prep.json"
        echo "Running preprocessing for: $CONFIG_PATH"
        python3 $SCRIPT_PATH/run_preprocessing.py --config "$CONFIG_PATH"
    done
done