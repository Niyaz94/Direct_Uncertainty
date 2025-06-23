#!/bin/bash -l


DATASET_VALUES=("BRAIN_TUMOR_TASK1" "PROSTATE_TASK1" "PROSTATE_TASK2")

MODEL_VALUES=("ensemble" "bayesian" "hier_probabilistic" "probabilistic")
MULTIPLE_VALUES=("MULTIPLE" "SINGLE" "UNCERTAINTY_CLASS")



for DATASET in "${DATASET_VALUES[@]}"; do
    for TYPE in "${MULTIPLE_VALUES[@]}"; do
        for MODEL in "${MODEL_VALUES[@]}"; do
            sbatch script/run_predict.sh $DATASET $MODEL $TYPE "test"         
        done
    done
done

