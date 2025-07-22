#!/bin/bash -l


#SBATCH -J MyJobNZJL

#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=48:00:00 

#SBATCH --gres=gpu
#SBATCH --array=0-4


## RAN_DATE=$(date +%s)
## exec > "output/out_${RAN_DATE}.out" 2> "error/err_${RAN_DATE}.err"

#SBATCH --output="/U-Net/log/output/out_%j.out"
#SBATCH --error="/U-Net/log/error/err_%j.err"

## transition to the directory from which sbatch was called
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

module add monai/1.2.0

source /U-Net/unet-env/bin/activate

SCRIPT_PATH="/U-Net"



# Check if SLURM_ARRAY_TASK_ID is set, otherwise default to 0
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    FILE_NUMBER=0  # Default value
else
    FILE_NUMBER=$SLURM_ARRAY_TASK_ID
fi


DATASET_VALUES=("BRAIN_TUMOR" "BRAIN_TUMOR_TASK1" "KIDNEY" "KNEE" "PROSTATE" "PROSTATE_TASK1" "PROSTATE_TASK2" "BRAIN_GROWTH" "PANCREAS" "SIJ" "LUNG" "HEART" "PANCREATIC_LESION")
MODEL_VALUES=("ensemble" "bayesian" "hier_probabilistic" "probabilistic")
MULTIPLE_VALUES=("SINGLE" "MULTIPLE" "UNCERTAINTY_CLASS")

DATASET_NAME=${1^^}  # First argument passed to sbatch
if [[ -z "$DATASET_NAME"  || ! " ${DATASET_VALUES[*]} " =~ [[:space:]]${DATASET_NAME}[[:space:]] ]]; then
    echo "Error: Please provide Dataset name. The value should be: BRAIN_TUMOR | BRAIN_TUMOR_TASK1 | KIDNEY | KNEE | PROSTATE | PROSTATE_TASK1 | PROSTATE_TASK2 | BRAIN_GROWTH | PANCREAS | SIJ | LUNG | HEART | PANCREATIC_LESION"
    exit 1
fi

MODEL_NAME=${2,,}  # making the input LowerCase
if [[ -z "$MODEL_NAME" || ! " ${MODEL_VALUES[*]} " =~ [[:space:]]${MODEL_NAME}[[:space:]] ]]; then
    echo "Error: Please provide Model name. The value should be: ensemble | bayesian | hier_probabilistic | probabilistic"
    exit 1
fi

MODEL_TYPE=${3^^} # making the input UpperCase
if [[ -z "$MODEL_TYPE" || ! " ${MULTIPLE_VALUES[*]} " =~ [[:space:]]${MODEL_TYPE}[[:space:]] ]]; then
    echo "Error: Please provide Model type. The value should be SINGLE | MULTIPLE | UNCERTAINTY_CLASS"
    exit 1
fi

PREDICTION_TYPE=${4,,}
if [[ -z "$PREDICTION_TYPE" ]]; then
    echo "Error: Please provide Prediction type. The value should be test | train"
    exit 1
fi


if [[ "$PREDICTION_TYPE" == "test" ]]; then
    python3 $SCRIPT_PATH/run_predict.py --config $SCRIPT_PATH/configs/$DATASET_NAME/$MODEL_TYPE/$MODEL_NAME/config_pred_$FILE_NUMBER.json
elif [ "$PREDICTION_TYPE" == "train" ]; then
    python3 $SCRIPT_PATH/run_predict.py --config $SCRIPT_PATH/configs/$DATASET_NAME/$MODEL_TYPE/$MODEL_NAME/config_tpred_$FILE_NUMBER.json
else
    echo "Wrong Prediction Type Has been provided!!! the value shoud be either [train|test]"
fi

