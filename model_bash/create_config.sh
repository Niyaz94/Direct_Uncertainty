#!/bin/bash -l

# ./script/create_config.sh train
# ./script/create_config.sh predict
# ./script/create_config.sh train_predict

source /U-Net/unet-env/bin/activate

SCRIPT_PATH="/U-Net"


CONFIG_TYPE=${1,,}
if [[ -z "$CONFIG_TYPE" ]]; then
    echo "Error: DO you need to create config files for training or predicting? the value should be [train|predict|train_predict]"
    exit 1
fi
echo $CONFIG_TYPE

# DATASET_VALUES=("BRAIN_TUMOR" "KIDNEY" "KNEE" "PROSTATE" "BRAIN_GROWTH" "PANCREAS" "SIJ" "LUNG" "HEART" "PANCREATIC_LESION")
# DATASET_VALUES=("HEART" "KIDNEY" "KNEE" "PROSTATE" "BRAIN_GROWTH" "SIJ" "PANCREAS" "LUNG" "PANCREATIC_LESION")
DATASET_VALUES=("BRAIN_TUMOR_TASK1" "PROSTATE_TASK1" "PROSTATE_TASK2")

# DATASET_SIZE=("2D" "2D" "2D" "2D" "2D" "2D" "3D" "3D" "3D")
DATASET_SIZE=("2D" "2D" "2D")

MODEL_VALUES=("ensemble" "bayesian" "hier_probabilistic" "probabilistic")
# MODEL_VALUES=("hier_probabilistic")

MULTIPLE_VALUES=("single" "multiple" "uncertainty_class")
# MULTIPLE_VALUES=("uncertainty_class")

for INDEX in "${!DATASET_VALUES[@]}"; do
    for TYPE in "${MULTIPLE_VALUES[@]}"; do
        for MODEL in "${MODEL_VALUES[@]}"; do
            if [[ "$CONFIG_TYPE" == "train" ]]; then
                python3 $SCRIPT_PATH/utility/create_config.py --model-name $MODEL --model-type $TYPE  --dataset-name "${DATASET_VALUES[$INDEX]}" --output-file config_train --dataset-shape "${DATASET_SIZE[$INDEX]}" --epoch-number 5000 
            elif [[ "$CONFIG_TYPE" == "predict" ]]; then
                python3 $SCRIPT_PATH/utility/create_config_pred.py --model-name $MODEL --model-type $TYPE  --dataset-name "${DATASET_VALUES[$INDEX]}" --dataset-shape "${DATASET_SIZE[$INDEX]}"
            elif [[ "$CONFIG_TYPE" == "train_predict" ]]; then
                python3 $SCRIPT_PATH/utility/create_config_pred.py --model-name $MODEL --model-type $TYPE  --dataset-name "${DATASET_VALUES[$INDEX]}" --dataset-shape "${DATASET_SIZE[$INDEX]}"  --pred-type "train"
            else
                echo "The config type should be either [train|predict]"
                break 3
            fi

        done
    done
done