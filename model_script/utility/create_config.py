
import os
import json
import argparse
from pathlib import Path
from enum import  Enum


# python3 create_config.py --model-name [bayesian|probabilistic|hier_probabilistic] --dataset-name heart --output-file config_train --dataset-shape 2D --epoch-number 1000

def create_config(ROOT_PATH,MODEL_NAME,DATASET_NAME,MODEL_TYPE,LOG_FILE_NAME,EPOCH_NUMBER,JSON_FILE_PATH_NAME,DATASET_SHAPE):
    if MODEL_NAME=="probabilistic":
        ARCHITECTURE_NAME="Probabilistic_UNet"
    elif MODEL_NAME=="hier_probabilistic":
        ARCHITECTURE_NAME="Hierarchical_Prob_UNet"
    elif MODEL_NAME=="bayesian":
        ARCHITECTURE_NAME="Bayesian_UNet"
    elif MODEL_NAME=="ensemble":
        ARCHITECTURE_NAME="Generic_UNet"
    else:
        raise NotImplementedError("You provided the wrong model name.")
    
    if DATASET_SHAPE=="2D":
        NETWORK_TYPE="2d"
    elif DATASET_SHAPE=="3D":
        NETWORK_TYPE="3d_fullres"
    else:
        raise NotImplementedError("You provided the wrong dataset shape.")
    
    Path(f"{ROOT_PATH}/models/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/").mkdir(parents=True, exist_ok=True)            
    Path(f"{ROOT_PATH}/predictions/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/").mkdir(parents=True, exist_ok=True)   
    Path(f"{ROOT_PATH}/results/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/loss").mkdir(parents=True, exist_ok=True)   
    
    
    preprocessed_directory = Path(f"{ROOT_PATH}/data/{DATASET_NAME}/{MODEL_TYPE}/preprocessed")
    
    plans_file_path = [file.name for file in preprocessed_directory.glob("nnUNet*.pkl")]
    
    # print(f"{ROOT_PATH}/data/{DATASET_NAME}/{MODEL_TYPE}/preprocessed",plans_file_path)
    if len(plans_file_path) > 1:
        raise Exception(f"More than one .pkl file found: {plans_file_path}")

    folder_with_preprocessed_data = [file.name for file in preprocessed_directory.glob("*_stage0")]
    if len(folder_with_preprocessed_data) > 1:
        raise Exception(f"More than one .pkl file found: {folder_with_preprocessed_data}")
  
    for index in range(0,5):
        train_config= {
            "fold": index,
            "random_state": 12345,
            "plans_file_path": f"{ROOT_PATH}/data/{DATASET_NAME}/{MODEL_TYPE}/preprocessed/{plans_file_path[0]}",
            "folder_with_preprocessed_data": f"{ROOT_PATH}/data/{DATASET_NAME}/{MODEL_TYPE}/preprocessed/{folder_with_preprocessed_data[0]}",
            "deep_supervision":False,
            "output_folder": f"{ROOT_PATH}/models/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/",
            "log_file": f"{ROOT_PATH}/results/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/loss/fold{index}_{LOG_FILE_NAME}.txt", 
            "network_type": NETWORK_TYPE,
            "architecture_name": ARCHITECTURE_NAME,
            "encoder_type":"resnet34",
            "initial_lr": 0.1,
            "momentum": 0.99,
            "nesterov": True,
            "numOfEpochs": EPOCH_NUMBER,
            "tr_batches_per_epoch": 250,
            "val_batches_per_epoch": 50,
            "checkpoint_frequency": 10,
            "continue_training":False, 
            "checkpoint": f"{ROOT_PATH}/models/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/fold_{index}_model_best.model",
            "num_of_samples": 10,
            "save_dir": f"{ROOT_PATH}/predictions/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/"
        }
        
        directory = os.path.join(ROOT_PATH, "configs", DATASET_NAME,MODEL_TYPE, MODEL_NAME)
        os.makedirs(directory, exist_ok=True)
        
        with open(f"{directory}/{JSON_FILE_NAME}_{index}.json", "w") as json_file:
            json.dump(train_config, json_file, indent=4)  # indent=4 for pretty formatting


class MODEL_TYPE_ENUM(Enum):
    single    = "single"
    multiple    = "multiple"
    uncertainty_class    = "uncertainty_class"
    
    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with parameters.")
    
    parser.add_argument("--model-name",     type=str, required=True,  help="Model Name")
    parser.add_argument("--dataset-name",   type=str, required=True,  help="Dataset Name")
    
    parser.add_argument("--dataset-shape",  type=str, required=False, help="Dataset Shape",default="2D") 
    parser.add_argument("--log-file-name",  type=str, required=False, default="log",  help="Log File Name") 
    parser.add_argument("--root-path",      type=str, required=False, help="Root Path",default="/U-Net")
    parser.add_argument("--output-file",    type=str, required=False,   help="The path for output config",default="config_train.json")
    parser.add_argument("--epoch-number",   type=int, required=False, default=2000,  help="Dataset Name")
    parser.add_argument('--model-type',     type=MODEL_TYPE_ENUM, choices=list(MODEL_TYPE_ENUM),default=MODEL_TYPE_ENUM.single)
    
    try:
        args = parser.parse_args()

        MODEL_NAME      = args.model_name
        DATASET_NAME    = args.dataset_name.upper()
        MODEL_TYPE      = args.model_type.value.upper()
        ROOT_PATH       = args.root_path
        LOG_FILE_NAME   = args.log_file_name
        EPOCH_NUMBER    = args.epoch_number
        DATASET_SHAPE   = args.dataset_shape
        JSON_FILE_NAME  = args.output_file
        
        # print(DATASET_NAME,MODEL_TYPE)
        
        create_config(ROOT_PATH,MODEL_NAME,DATASET_NAME,MODEL_TYPE,LOG_FILE_NAME,EPOCH_NUMBER,JSON_FILE_NAME,DATASET_SHAPE)
        

    except SystemExit as e:
        print("Error: Missing required arguments!")
        print("Usage: script.py --model-name <name> --dataset-name <dataset>")
        raise e