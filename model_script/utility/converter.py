import glob
import shutil
import os
import nibabel as nib
import numpy as np
import json
#from pycimg import CImg

SEED = 42
np.random.seed(SEED)
train_split = 0.8

multichannel = {"HEART" :-1, "KIDNEY": -1, "BRAIN_TUMOR": 2, "KNEE": -1,"PANCREATIC_LESION": -1,
              "PROSTATE":-1, "BRAIN_GROWTH":-1,"PANCREAS":-1,"SIJ":-1,"LUNG":-1}
dimensions = {"HEART" :"2d", "KIDNEY": "2d", "BRAIN_TUMOR": "2d", "KNEE": "2d","PANCREATIC_LESION": "3d_fullres",
              "PROSTATE":"2d", "BRAIN_GROWTH":"2d","PANCREAS":"3d_fullres","SIJ":"2d","LUNG":"3d_fullres"}
squeeze = {"HEART" :True, "KIDNEY": False, "BRAIN_TUMOR": False, "KNEE": True,"PANCREATIC_LESION": False,
              "PROSTATE":True, "BRAIN_GROWTH":False,"PANCREAS":False,"SIJ":True,"LUNG":False}

# HEART = (512, 512, 1)
# KIDNEY (497, 497)
# BRAIN_TUMOR (240, 240, 4)
# KNEE (448, 448, 1)
# PANCREATIC_LESION (41, 512, 512)
# PROSTATE (960, 640, 1)
# BRAIN_GROWTH (256, 256)
# PANCREAS (100, 512, 512)
# SIJ (32, 41, 1)
# LUNG (72, 59, 10)


saveDir = '/U-Net/data/'
datasets = glob.glob('/DATASET/*')

datasets = ['/DATASET/PROSTATE']

dataset_case_name='/task02_seg*.nii.gz'
dataset_postfix = '_TASK2'

for dataset in datasets:
    
    basename = os.path.basename(dataset)
    fname = saveDir + basename + dataset_postfix
    
    os.mkdir(fname)
    os.mkdir(fname + '/MULTIPLE')
    os.mkdir(fname + '/SINGLE')
    
    os.mkdir(fname + '/MULTIPLE/imagesTr')
    os.mkdir(fname + '/MULTIPLE/imagesTs')
    os.mkdir(fname + '/MULTIPLE/labelsTr')
    os.mkdir(fname + '/MULTIPLE/labelsTs')
    os.mkdir(fname + '/MULTIPLE/preprocessed')
    os.mkdir(fname + '/MULTIPLE/cropped')
    os.mkdir(fname + '/MULTIPLE/configs')
    os.mkdir(fname + '/MULTIPLE/MODELS')

    os.mkdir(fname + '/SINGLE/imagesTr')
    os.mkdir(fname + '/SINGLE/imagesTs')
    os.mkdir(fname + '/SINGLE/labelsTr')
    os.mkdir(fname + '/SINGLE/labelsTs')
    os.mkdir(fname + '/SINGLE/preprocessed')
    os.mkdir(fname + '/SINGLE/cropped')
    os.mkdir(fname + '/SINGLE/configs')
    os.mkdir(fname + '/SINGLE/MODELS')
    
    cases = sorted([int(os.path.basename(i)) for i in glob.glob(dataset+"/*")])
    
    np.random.shuffle(cases)
    SPLIT = int(len(cases)*train_split)
    TRAIN_IDS = cases[:SPLIT]
    TEST_IDS =  cases[SPLIT:]

    dataDir1 = {
        "name": "DUM",
        "description": "DUM",
        "reference": "DUM",
        "licence":"NA",
        "relase":"NA",
        "tensorImageSize": None,
        "modality": None,
        "labels": None,
        "numTraining": 0,
        "numTest": 0,
        "training":[],
        "test": []
    }
    
    dataDir2 = {
        "name": "DUM",
        "description": "DUM",
        "reference": "DUM",
        "licence":"NA",
        "relase":"NA",
        "tensorImageSize": None,
        "modality": None,
        "labels": None,
        "numTraining": 0,
        "numTest": 0,
        "training":[],
        "test": []
    }

    print(basename)
    if "2d" in dimensions[basename]:
        dataDir1["tensorImageSize"] = "2D"
        dataDir2["tensorImageSize"] = "2D"
    else:
        dataDir1["tensorImageSize"] = "3D"
        dataDir2["tensorImageSize"] = "3D"

    num1 = 0
    num2 = 0
    for ntrain_id in TRAIN_IDS:
        print('#################')
        print(ntrain_id)
        fname = dataset + '/' + str(ntrain_id) + '/image.nii.gz'
        im = nib.load(fname).get_fdata()
        aff = nib.load(fname).affine

        if squeeze[basename]:
            im = np.squeeze(im)

        if multichannel[basename] > 0:
            channels = [im.take(i, axis=multichannel[basename]) for i in range(im.shape[multichannel[basename]])]
        else:
            channels = [im]

        if dimensions[basename] == '2d':
            channels = [np.reshape(c,c.shape + (1,)) for c in channels]

        # for ch in channels:
        #     print(ch.shape)
        #     CImg(ch).display("channel");
        #  '/task01_seg*.nii.gz'
        snames = sorted(glob.glob(dataset + '/' + str(ntrain_id) + dataset_case_name))
        tasks = sorted(list(set([os.path.basename(s).split('_')[0] for s in snames])))
        cases = sorted(list(set([os.path.basename(s).split('_')[1].split('.')[0] for s in snames])))

        dataDir1['labels'] = {"0": "background"} | {str(v):k for k,v in zip(tasks,range(1,len(tasks)+1))}
        dataDir2['labels'] = {"0": "background"} | {str(v):k for k,v in zip(tasks,range(1,len(tasks)+1))}

        dataDir1['modality'] = { str(v): "channel_"+str(v) for v in range(len(channels))}
        dataDir2['modality'] = { str(v): "channel_"+str(v) for v in range(len(channels))}
        
        # print(tasks)
        # print(cases)

        segmentations = []
        for case in cases:
            # print('*******************')
            # print(case)
            segmentation = np.zeros(channels[0].shape,dtype=np.uint8)
            flag = False
            for ntask,task in enumerate(tasks):
                sname = dataset + '/' + str(ntrain_id) + '/' + task + '_' + case + '.nii.gz'
                if os.path.isfile(sname):
                    flag = True
                    seg = nib.load(sname).get_fdata()
                    if squeeze[basename]:
                        seg = np.squeeze(seg)
                    if dimensions[basename] == '2d':
                        seg = np.reshape(seg,seg.shape + (1,))
                    segmentation[seg != 0] = ntask + 1

                    # CImg(seg).display(case + ' ' + task);
                    # print(ntask,task,np.unique(seg))
                          
            if flag:
                segmentations.append(segmentation)
                # print(segmentations[-1].shape,np.unique(segmentations[-1]))

        for nseg,segmentation in enumerate(segmentations):
            item = {}
            for nchannel,channel in enumerate(channels):
                savename = saveDir + basename + dataset_postfix + '/MULTIPLE/imagesTr/case_' + str(ntrain_id) + '_' + str(nseg) + '_000' + str(nchannel) + '.nii.gz'
                niftiImage = nib.Nifti1Image(channel, affine=aff)
                nib.save(niftiImage,savename)
            savename = saveDir + basename + dataset_postfix + '/MULTIPLE/labelsTr/case_' + str(ntrain_id) + '_' + str(nseg) + '.nii.gz'
            niftiImage = nib.Nifti1Image(segmentation, affine=aff)
            nib.save(niftiImage,savename)
            item["image"] = "./imagesTr/" + os.path.basename(savename)
            item["label"] = "./labelsTr/" + os.path.basename(savename)
            dataDir1['training'].append(item)
            num1 += 1

        item = {}
        nseg = np.random.randint(0,len(segmentations))
        for nchannel,channel in enumerate(channels):
            savename = saveDir + basename + dataset_postfix + '/SINGLE/imagesTr/case_' + str(ntrain_id) + '_' + str(nseg) + '_000' + str(nchannel) + '.nii.gz'
            niftiImage = nib.Nifti1Image(channel, affine=aff)
            nib.save(niftiImage,savename)
        savename = saveDir + basename + dataset_postfix + '/SINGLE/labelsTr/case_' + str(ntrain_id) + '_' + str(nseg) + '.nii.gz'
        niftiImage = nib.Nifti1Image(segmentations[nseg], affine=aff)
        nib.save(niftiImage,savename)
        item["image"] = "./imagesTr/" + os.path.basename(savename)
        item["label"] = "./labelsTr/" + os.path.basename(savename)
        dataDir2['training'].append(item)
        num2 += 1

    dataDir1['numTraining'] = num1
    dataDir2['numTraining'] = num2                

    f = open(saveDir + basename + dataset_postfix + '/MULTIPLE/dataset.json','w')
    json.dump(dataDir1,f,indent=4)
    f.close()
    
    f = open(saveDir + basename + dataset_postfix + '/SINGLE/dataset.json','w')
    json.dump(dataDir2,f,indent=4)
    f.close()

    for ntrain_id in TEST_IDS:
        print('#################')
        print(ntrain_id)
        fname = dataset + '/' + str(ntrain_id) + '/image.nii.gz'
        im = nib.load(fname).get_fdata()
        aff = nib.load(fname).affine

        if squeeze[basename]:
            im = np.squeeze(im)

        if multichannel[basename] > 0:
            channels = [im.take(i, axis=multichannel[basename]) for i in range(im.shape[multichannel[basename]])]
        else:
            channels = [im]

        if dimensions[basename] == '2d':
            channels = [np.reshape(c,c.shape + (1,)) for c in channels]
        
        # 01_seg*.nii.gz
        snames = sorted(glob.glob(dataset + '/' + str(ntrain_id) + dataset_case_name))
        tasks = sorted(list(set([os.path.basename(s).split('_')[0] for s in snames])))
        cases = sorted(list(set([os.path.basename(s).split('_')[1].split('.')[0] for s in snames])))

        segmentations = []
        for case in cases:
            # print('*******************')
            # print(case)
            segmentation = np.zeros(channels[0].shape,dtype=np.uint8)
            flag = False
            for ntask,task in enumerate(tasks):
                sname = dataset + '/' + str(ntrain_id) + '/' + task + '_' + case + '.nii.gz'
                if os.path.isfile(sname):
                    flag = True
                    seg = nib.load(sname).get_fdata()
                    if squeeze[basename]:
                        seg = np.squeeze(seg)
                    if dimensions[basename] == '2d':
                        seg = np.reshape(seg,seg.shape + (1,))
                    segmentation[seg != 0] = ntask + 1
                          
            if flag:
                segmentations.append(segmentation)
                # print(segmentations[-1].shape,np.unique(segmentations[-1]))

        for nseg,segmentation in enumerate(segmentations):
            for nchannel,channel in enumerate(channels):
                savename = saveDir + basename + dataset_postfix + '/MULTIPLE/imagesTs/case_' + str(ntrain_id) + '_' + str(nseg) + '_000' + str(nchannel) + '.nii.gz'
                niftiImage = nib.Nifti1Image(channel, affine=aff)
                nib.save(niftiImage,savename)
                savename = saveDir + basename + dataset_postfix +  '/SINGLE/imagesTs/case_' + str(ntrain_id) + '_' + str(nseg) + '_000' + str(nchannel) + '.nii.gz'
                niftiImage = nib.Nifti1Image(channel, affine=aff)
                nib.save(niftiImage,savename)
            savename = saveDir + basename + dataset_postfix + '/MULTIPLE/labelsTs/case_' + str(ntrain_id) + '_' + str(nseg) + '.nii.gz'
            niftiImage = nib.Nifti1Image(segmentation, affine=aff)
            nib.save(niftiImage,savename)
            savename = saveDir + basename + dataset_postfix + '/SINGLE/labelsTs/case_' + str(ntrain_id) + '_' + str(nseg) + '.nii.gz'
            niftiImage = nib.Nifti1Image(segmentation, affine=aff)
            nib.save(niftiImage,savename)

    config_prep = {
        "default_num_threads": 8,
        "dont_run_preprocessing": False,
        "tl": 8,
        "tf": 8,
        "planner_name3d": None,
        "planner_name2d": None,
        "verify_dataset_integrity_flag": True,
        "random_state": 12345,
        "preprocessing_output_dir": None,
        "folder_with_raw_data": None,
        "folder_with_cropped_data": None
    }

    if dimensions[basename] == '2d':
        config_prep['planner_name2d'] = "ExperimentPlanner2D_v21"
    else:
        config_prep['planner_name3d'] = "ExperimentPlanner3D_v21"
    config_prep["folder_with_raw_data"] = saveDir + basename + dataset_postfix + '/MULTIPLE/'
    config_prep["preprocessing_output_dir"] = saveDir + basename + dataset_postfix + '/MULTIPLE/preprocessed'
    config_prep["folder_with_cropped_data"] = saveDir + basename + dataset_postfix + '/MULTIPLE/cropped/'

    f = open(saveDir + basename + dataset_postfix + '/MULTIPLE/configs/config_prep.json','w')
    json.dump(config_prep,f,indent=4)
    f.close()

    config_prep["folder_with_raw_data"] = saveDir + basename + dataset_postfix + '/SINGLE/'
    config_prep["preprocessing_output_dir"] = saveDir + basename + dataset_postfix + '/SINGLE/preprocessed'
    config_prep["folder_with_cropped_data"] = saveDir + basename + dataset_postfix + '/SINGLE/cropped/'

    f = open(saveDir + basename + dataset_postfix + '/SINGLE/configs/config_prep.json','w')
    json.dump(config_prep,f,indent=4)
    f.close()

    config_train = {
      "fold": None,
      "random_state": 12345,
      "plans_file_path": None,
      "folder_with_preprocessed_data": None,
      "deep_supervision": True,
    
      "output_folder": None,
      "log_file": None, 
    
      "network_type": dimensions[basename],
      "architecture_name": "Generic_UNet",
      "initial_lr": 0.01,
      "momentum": 0.99,
      "nesterov": True,
    
      "numOfEpochs": 1000,
      "tr_batches_per_epoch": 250,
      "val_batches_per_epoch": 50,
      "checkpoint_frequency": 10,
      "continue_training":False, 
      "checkpoint": "",
    }

    for fold in range(5):
        config_train['fold'] = fold
        config_train["output_folder"] = saveDir + basename + dataset_postfix + '/MULTIPLE/MODELS/'
        config_train["log_file"] = saveDir + basename + dataset_postfix + '/MULTIPLE/MODELS/log' + str(fold) + '.txt'
        if dimensions[basename] == '2d':
            config_train["plans_file_path"] = saveDir + basename + dataset_postfix + '/MULTIPLE/preprocessed/nnUNetPlansv2.1_plans_2D.pkl'
            config_train["folder_with_preprocessed_data"] = saveDir + basename + dataset_postfix + '/MULTIPLE/preprocessed/nnUNetData_plans_v2.1_2D_stage0'
        else:
            config_train["plans_file_path"] = saveDir + basename + dataset_postfix+ '/MULTIPLE/preprocessed/nnUNetPlansv2.1_plans_3D.pkl'
            config_train["folder_with_preprocessed_data"] = saveDir + basename + dataset_postfix + '/MULTIPLE/preprocessed/nnUNetData_plans_v2.1_stage0'
            
        f = open(saveDir + basename + dataset_postfix + '/MULTIPLE/configs/config_train_' + str(fold) + '.json','w')
        json.dump(config_train,f,indent=4)
        f.close()
        
        config_train["output_folder"] = saveDir + basename + dataset_postfix + '/SINGLE/MODELS/'
        config_train["log_file"] = saveDir + basename + dataset_postfix + '/SINGLE/MODELS/log' + str(fold) + '.txt'
        if dimensions[basename] == '2d':
            config_train["plans_file_path"] = saveDir + basename + dataset_postfix + '/SINGLE/preprocessed/nnUNetPlansv2.1_plans_2D.pkl'
            config_train["folder_with_preprocessed_data"] = saveDir + basename + dataset_postfix + '/SINGLE/preprocessed/nnUNetData_plans_v2.1_2D_stage0'
        else:
            config_train["plans_file_path"] = saveDir + basename + dataset_postfix + '/SINGLE/preprocessed/nnUNetPlansv2.1_plans_3D.pkl'
            config_train["folder_with_preprocessed_data"] = saveDir + basename + dataset_postfix + '/SINGLE/preprocessed/nnUNetData_plans_v2.1_stage0'
            
        f = open(saveDir + basename + dataset_postfix + '/SINGLE/configs/config_train_' + str(fold) + '.json','w')
        json.dump(config_train,f,indent=4)
        f.close()
    

