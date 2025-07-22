#!/bin/bash -l


#SBATCH -J MyJobNZJL

#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=48:00:00 
#SBATCH --gres=gpu


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




DATASET_VALUES=("BRAIN_TUMOR" "KIDNEY" "KNEE" "PROSTATE" "BRAIN_GROWTH" "PANCREAS" "SIJ" "LUNG" "HEART" "PANCREATIC_LESION")
MODEL_VALUES=("ensemble" "bayesian" "hier_probabilistic" "probabilistic")
MULTIPLE_VALUES=("SINGLE" "MULTIPLE")

DATASET_NAME=${1^^}  # First argument passed to sbatch
if [[ -z "$DATASET_NAME"  || ! " ${DATASET_VALUES[*]} " =~ [[:space:]]${DATASET_NAME}[[:space:]] ]]; then
    echo "Error: Please provide Dataset name. The value should be: BRAIN_TUMOR | KIDNEY | KNEE | PROSTATE | BRAIN_GROWTH | PANCREAS | SIJ | LUNG | HEART | PANCREATIC_LESION"
    exit 1
fi


MODEL_NAME=${2,,}  # making the input LowerCase
if [[ -z "$MODEL_NAME" || ! " ${MODEL_VALUES[*]} " =~ [[:space:]]${MODEL_NAME}[[:space:]] ]]; then
    echo "Error: Please provide Model name. The value should be: ensemble | bayesian | hier_probabilistic | probabilistic"
    exit 1
fi

MODEL_TYPE=${3^^} # making the input UpperCase
if [[ -z "$MODEL_TYPE" || ! " ${MULTIPLE_VALUES[*]} " =~ [[:space:]]${MODEL_TYPE}[[:space:]] ]]; then
    echo "Error: Please provide Model type. The value should be SINGLE | MULTIPLE"
    exit 1
fi

FOLD_NUMBER=$4 # fold number
if [[ -z "$FOLD_NUMBER" ]]; then
    echo "Error: You should provide a fold number the value is between [0-4]"
    exit 1
fi


# echo "${DATASET_NAME}-${MODEL_TYPE}-${MODEL_NAME}"
#  sbatch script/run.sh KIDNEY bayesian single
if [[ "$MODEL_NAME" == "bayesian" || "$MODEL_NAME" == "ensemble" ]]; then
    # echo "Bayesian Model: ${DATASET_NAME}-${MODEL_TYPE}-${MODEL_NAME}"
    python3 $SCRIPT_PATH/run_training.py --config $SCRIPT_PATH/configs/$DATASET_NAME/$MODEL_TYPE/$MODEL_NAME/config_train_$FOLD_NUMBER.json
# elif [ "$MODEL_NAME" == "ensemble" ]; then
    # echo "Ensemble Model: ${DATASET_NAME}-${MODEL_TYPE}-${MODEL_NAME}"
    # echo "Not Implemented Yet"
else
    # echo "Probabilistic Model: ${DATASET_NAME}-${MODEL_TYPE}-${MODEL_NAME}"
    python3 $SCRIPT_PATH/run_training_prob.py --config $SCRIPT_PATH/configs/$DATASET_NAME/$MODEL_TYPE/$MODEL_NAME/config_train_$FOLD_NUMBER.json
fi