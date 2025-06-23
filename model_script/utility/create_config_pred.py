
import os
import json
import argparse
from pathlib import Path
from enum import  Enum


# python3 utility/create_config_pred.py --model-name [bayesian|probabilistic|hier_probabilistic] --model-type [single|multiple] --dataset-name heart --output-file config_pred --dataset-shape 2D
# python3 utility/create_config_pred.py --model-name bayesian --model-type single --dataset-name heart  --dataset-shape 2D
# python3 utility/create_config_pred.py --model-name bayesian --model-type single --dataset-name heart  --dataset-shape 2D --pred-type train


def create_config(ROOT_PATH,MODEL_NAME,DATASET_NAME,MODEL_TYPE,JSON_FILE_NAME,DATASET_SHAPE,PRED_TYPE):
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
    
    if PRED_TYPE=="test":
        FOLDER_NAME="imagesTs"
        FOLDER_INDEX=""
    elif PRED_TYPE=="train":
        FOLDER_NAME="imagesTr"
        FOLDER_INDEX="t_"
        JSON_FILE_NAME="config_tpred"
    else:
        raise NotImplementedError("You provided the wrong prediction type.")
    
    
    preprocessed_directory = Path(f"{ROOT_PATH}/data/{DATASET_NAME}/{MODEL_TYPE}/preprocessed")
    
    plans_file_path = [file.name for file in preprocessed_directory.glob("nnUNet*.pkl")]
    
    # print(f"{ROOT_PATH}/data/{DATASET_NAME}/{MODEL_TYPE}/preprocessed",plans_file_path)
    if len(plans_file_path) > 1:
        raise Exception(f"More than one .pkl file found: {plans_file_path}")

    folder_with_preprocessed_data = [file.name for file in preprocessed_directory.glob("*_stage0")]
    if len(folder_with_preprocessed_data) > 1:
        raise Exception(f"More than one .pkl file found: {folder_with_preprocessed_data}")
  
    for index in range(0,5):
        
        result_fold_directory=Path(f"{ROOT_PATH}/predictions/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/{FOLDER_INDEX}fold{index}") 
        result_fold_directory.mkdir(parents=True, exist_ok=True)
        os.chmod(result_fold_directory, 0o770)
        
        train_config= {
            "network_type": NETWORK_TYPE,
            "architecture_name": ARCHITECTURE_NAME,
            "encoder_type":"resnet34",
            "disable_tta": False,
            "input_folder": f"{ROOT_PATH}/data/{DATASET_NAME}/{MODEL_TYPE}/{FOLDER_NAME}",
            "output_folder": f"{ROOT_PATH}/predictions/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/{FOLDER_INDEX}fold{index}",
            "plans_file_path": f"{ROOT_PATH}/data/{DATASET_NAME}/{MODEL_TYPE}/preprocessed/{plans_file_path[0]}",
            "checkpoints":{
                "1": f"{ROOT_PATH}/models/{DATASET_NAME}/{MODEL_TYPE}/{MODEL_NAME}/fold_{index}_model_best.model",
            }
        }
        
        directory = os.path.join(ROOT_PATH, "configs", DATASET_NAME,MODEL_TYPE, MODEL_NAME)
        os.makedirs(directory, exist_ok=True)
        
        with open(f"{directory}/{JSON_FILE_NAME}_{index}.json", "w") as json_file:
            json.dump(train_config, json_file, indent=4)  # indent=4 for pretty formatting


class MODEL_TYPE_ENUM(Enum):
    single              = "single"
    multiple            = "multiple"
    uncertainty_class   = "uncertainty_class"
    
    
    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]
    
class PRED_TYPE_ENUM(Enum):
    test    = "test"
    train   = "train"
    
    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with parameters.")
    
    parser.add_argument("--model-name",     type=str, required=True,  help="Model Name")
    parser.add_argument("--dataset-name",   type=str, required=True,  help="Dataset Name")
    parser.add_argument("--dataset-shape",  type=str, required=False, help="Dataset Shape",default="2D") 
    parser.add_argument("--root-path",      type=str, required=False, help="Root Path",default="/U-Net")
    parser.add_argument("--output-file",    type=str, required=False,   help="The path for output config",default="config_pred")
    parser.add_argument('--model-type',     type=MODEL_TYPE_ENUM, choices=list(MODEL_TYPE_ENUM),default=MODEL_TYPE_ENUM.single)
    parser.add_argument('--pred-type',      type=PRED_TYPE_ENUM , choices=list(PRED_TYPE_ENUM) ,default=PRED_TYPE_ENUM.test)
    
    try:
        args = parser.parse_args()

        MODEL_NAME      = args.model_name
        DATASET_NAME    = args.dataset_name.upper()
        MODEL_TYPE      = args.model_type.value.upper()
        ROOT_PATH       = args.root_path
        DATASET_SHAPE   = args.dataset_shape
        JSON_FILE_NAME  = args.output_file
        PRED_TYPE       = args.pred_type.value
        
        # print(PRED_TYPE)
                
        create_config(ROOT_PATH,MODEL_NAME,DATASET_NAME,MODEL_TYPE,JSON_FILE_NAME,DATASET_SHAPE,PRED_TYPE)
        

    except SystemExit as e:
        print("Error: Missing required arguments!")
        print("Usage: script.py --model-name <name> --dataset-name <dataset>")
        raise e