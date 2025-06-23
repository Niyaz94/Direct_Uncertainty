import argparse
import torch
import nibabel as nib
from copy import deepcopy
from typing import Tuple, Union, List
import numpy as np
import json
import pickle
import os




from batchgenerators.utilities.file_and_folder_operations import join, subfiles, maybe_mkdir_p,load_pickle
from batchgenerators.utilities.file_and_folder_operations import *

from c_unet.inference.predictor import Predictor



def predict_cases(
    plan_file,encoderType,architecture_name,checkpoints, stage, output_folder, list_of_lists, output_filenames, 
    do_tta=True, step_size=0.5,overwrite = False
):

    assert len(list_of_lists) == len(output_filenames)

    print('#################################################')
    print("emptying cuda cache")
    torch.cuda.empty_cache()

    trainer = Predictor(plan_file, encoderType, architecture_name,stage, False, True)
    trainer.process_plans(load_pickle(plan_file))

    trainer.output_folder = output_folder
    trainer.output_folder_base = output_folder
    trainer.initialize(False)

    params = [torch.load(i, map_location=torch.device('cpu')) for i in checkpoints]

    print('Trainer params')
    print(trainer.normalization_schemes, trainer.use_mask_for_norm,trainer.transpose_forward, trainer.intensity_properties)
    print('#####################################')

    if 'segmentation_export_params' in trainer.plans.keys():
        force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
        interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0


    print("starting prediction...")
    for list_of_list,output_filename in zip(list_of_lists, output_filenames):

        if os.path.isfile(output_filename) and not overwrite:
            continue

        d, _, dct = trainer.preprocess_patient(list_of_list)

        print("processing ", output_filename,d.shape,dct)
        print("predicting", output_filename)
        print('threeD',trainer.threeD)
        softmax = []
        for p in params:
            trainer.load_checkpoint_ram(p, False)
            #################################################################
            #d = d[0,0,:,:]
            #d = torch.tensor(d)
            #d.resize_((352,512))
            #d = d.reshape((1,1,352,512))
            #d = d.cuda()
            # out = trainer.network(d)
            #print('SHAPE',out.shape)
            ################################################################
            trainer_result=trainer.predict_preprocessed_data_return_seg_and_softmax(
                    d, do_tta, trainer.data_aug_params['mirror_axes'], True, step_size=step_size, 
                    use_gaussian=True, all_in_gpu=False,mixed_precision=True
            )
            softmax.append(trainer_result[1][None])

        softmax = np.vstack(softmax)
        softmax_mean = np.mean(softmax, 0)

        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        trainer.save_segmentation(softmax_mean, output_filename, dct, interpolation_order, region_class_order,None, None,None, None, force_separate_z, interpolation_order_z)



def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",required = True, help ="json file with inference configuration")

    args = parser.parse_args()
    config_file = args.config

    with open(config_file, 'r') as config_file:
        configs = json.load(config_file)

    input_folder = configs['input_folder']
    output_folder = configs['output_folder']
    disable_tta = configs['disable_tta']
    plan_file = configs['plans_file_path']
    try:
        encoderType = configs['encoder_type']
    except:
        encoderType = None
    checkpoints = list(configs['checkpoints'].values())

    maybe_mkdir_p(output_folder)

    # check input folder integrity
    expected_num_modalities = load_pickle(plan_file)['num_modalities']
    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

    # output file names
    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    # original input image names in a form of a list of lists, where j-th sublist contains n input image names for j-th case, these n inputs correspond to n Unet inputs 
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]

    # some configs
    with open(configs['plans_file_path'], 'rb') as plans_file:
        plans = pickle.load(plans_file)

    network_type = configs['network_type']
    assert network_type in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], "Incorrect network type!"

    possible_stages = list(plans['plans_per_stage'].keys())

    if (network_type == '3d_cascade_fullres' or network_type == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. Run 3d_fullres.")

    if network_type == '2d' or network_type == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    if 'architecture_name' in configs.keys():
        architecture_name = configs['architecture_name']
    else:
        architecture_name = "Generic_UNet"
        
    # start prediction
    step_size = 0.5
    predict_cases(plan_file,encoderType,architecture_name,checkpoints, stage, output_folder, list_of_lists, output_files, not disable_tta, step_size=step_size)


if __name__ == "__main__":
    main()
