import glob
import shutil
import os
import nibabel as nib
import numpy as np
import json


datasets = glob.glob('/U-Net/data/*')

datasets = ['/U-Net/data/HEART']

TH = 0.01

for fname in datasets:
    
    #os.mkdir(fname + '/UNCERTAINTY_CLASS')
    
    #os.mkdir(fname + '/UNCERTAINTY_CLASS/imagesTr')
    #os.mkdir(fname + '/UNCERTAINTY_CLASS/imagesTs')
    #os.mkdir(fname + '/UNCERTAINTY_CLASS/labelsTr')
    #os.mkdir(fname + '/UNCERTAINTY_CLASS/labelsTs')
    #os.mkdir(fname + '/UNCERTAINTY_CLASS/preprocessed')
    #os.mkdir(fname + '/UNCERTAINTY_CLASS/cropped')
    #os.mkdir(fname + '/UNCERTAINTY_CLASS/configs')

    modes = ['Tr','Ts']

    f = open(fname + '/MULTIPLE/dataset.json','r')
    data_dict = json.load(f)
    f.close()

    data_dict['training'] = []

    for mode in modes:
        cases = sorted(list(set(['_'.join(os.path.basename(f).split('_')[0:2]) for f in glob.glob(fname + '/MULTIPLE/images' + mode + '/*')])))
        print(cases)

        if mode =='Tr':
            data_dict['numTraining'] = len(cases)

        
        for case in cases:
            print(case)
            cname = fname + '/MULTIPLE/images' + mode + '/' + case + '_0_0000.nii.gz'
            dname = fname + '/UNCERTAINTY_CLASS/images' + mode + '/' + case + '_0_0000.nii.gz'
            shutil.copy(cname,dname)

            snames = glob.glob(fname + '/MULTIPLE/labels' + mode + '/' + case + '_*.nii.gz')
            imgs = []
            for sname in snames:
                im = nib.load(sname).get_fdata()
                aff = nib.load(sname).affine
                imgs.append(im)
            imgs = np.asarray(imgs)
            labels = np.unique(imgs)
            imgs = np.mean(imgs,axis=0)
            mean_labels = np.unique(imgs)

            new_labels = [i for i in mean_labels if np.min([abs(i-l) for l in labels]) > TH]

            max_label = np.max(labels) + 1
            for i in new_labels:
                imgs[imgs==i] = max_label

            savename = fname + '/UNCERTAINTY_CLASS/labels' + mode + '/' + case + '_0.nii.gz'
            niftiImage = nib.Nifti1Image(imgs, affine=aff)
            nib.save(niftiImage,savename)

            if mode == 'Tr':
                item = {}
                item['image'] = './imagesTr/' + case + '_0.nii.gz'
                item['label'] = './labelsTr/' + case + '_0.nii.gz'
                data_dict['training'].append(item)


        #    break
        #
        #break


    data_dict['labels'][str(int(max_label))] = 'uncertainty'

    f = open(fname + '/UNCERTAINTY_CLASS/dataset.json','w')
    json.dump(data_dict,f,indent=4)
    f.close()

    f = open(fname + '/MULTIPLE/configs/config_prep.json','r')
    config_prep = json.load(f)
    f.close()

    config_prep["preprocessing_output_dir"] = fname + '/UNCERTAINTY_CLASS/preprocessed'
    config_prep["folder_with_raw_data"] = fname + '/UNCERTAINTY_CLASS'
    config_prep["folder_with_cropped_data"] = fname + '/UNCERTAINTY_CLASS/cropped/'

    f = open(fname + '/UNCERTAINTY_CLASS/configs/config_prep.json','w')
    json.dump(config_prep,f,indent=4)
    f.close()
    



    
    """
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
           
        snames = sorted(glob.glob(dataset + '/' + str(ntrain_id) + '/task*.nii.gz'))
        tasks = sorted(list(set([os.path.basename(s).split('_')[0] for s in snames])))
        cases = sorted(list(set([os.path.basename(s).split('_')[1].split('.')[0] for s in snames])))

        dataDir1['labels'] = {"0": "background"} | {'"' + str(v) + '"':k for k,v in zip(tasks,range(1,len(tasks)+1))}
        dataDir2['labels'] = {"0": "background"} | {'"' + str(v) + '"':k for k,v in zip(tasks,range(1,len(tasks)+1))}

        dataDir1['modality'] = {'"' + str(v) + '"': "channel_"+str(v) for v in range(len(channels))}
        dataDir2['modality'] = {'"' + str(v) + '"': "channel_"+str(v) for v in range(len(channels))}
        
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
                savename = saveDir + basename + '/MULTIPLE/imagesTr/case_' + str(ntrain_id) + '_' + str(nseg) + '_000' + str(nchannel) + '.nii.gz'
                niftiImage = nib.Nifti1Image(channel, affine=aff)
                nib.save(niftiImage,savename)
            savename = saveDir + basename + '/MULTIPLE/labelsTr/case_' + str(ntrain_id) + '_' + str(nseg) + '.nii.gz'
            niftiImage = nib.Nifti1Image(segmentation, affine=aff)
            nib.save(niftiImage,savename)
            item["image"] = "./imagesTr/" + os.path.basename(savename)
            item["label"] = "./labelsTr/" + os.path.basename(savename)
            dataDir1['training'].append(item)
            num1 += 1

        item = {}
        nseg = np.random.randint(0,len(segmentations))
        for nchannel,channel in enumerate(channels):
            savename = saveDir + basename + '/SINGLE/imagesTr/case_' + str(ntrain_id) + '_' + str(nseg) + '_000' + str(nchannel) + '.nii.gz'
            niftiImage = nib.Nifti1Image(channel, affine=aff)
            nib.save(niftiImage,savename)
        savename = saveDir + basename + '/SINGLE/labelsTr/case_' + str(ntrain_id) + '_' + str(nseg) + '.nii.gz'
        niftiImage = nib.Nifti1Image(segmentations[nseg], affine=aff)
        nib.save(niftiImage,savename)
        item["image"] = "./imagesTr/" + os.path.basename(savename)
        item["label"] = "./labelsTr/" + os.path.basename(savename)
        dataDir2['training'].append(item)
        num2 += 1

    dataDir1['numTraining'] = num1
    dataDir2['numTraining'] = num2                

    f = open(saveDir + basename + '/MULTIPLE/dataset.json','w')
    json.dump(dataDir1,f,indent=4)
    f.close()
    
    f = open(saveDir + basename + '/SINGLE/dataset.json','w')
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
           
        snames = sorted(glob.glob(dataset + '/' + str(ntrain_id) + '/task*.nii.gz'))
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
                savename = saveDir + basename + '/MULTIPLE/imagesTs/case_' + str(ntrain_id) + '_' + str(nseg) + '_000' + str(nchannel) + '.nii.gz'
                niftiImage = nib.Nifti1Image(channel, affine=aff)
                nib.save(niftiImage,savename)
                savename = saveDir + basename + '/SINGLE/imagesTs/case_' + str(ntrain_id) + '_' + str(nseg) + '_000' + str(nchannel) + '.nii.gz'
                niftiImage = nib.Nifti1Image(channel, affine=aff)
                nib.save(niftiImage,savename)
            savename = saveDir + basename + '/MULTIPLE/labelsTs/case_' + str(ntrain_id) + '_' + str(nseg) + '.nii.gz'
            niftiImage = nib.Nifti1Image(segmentation, affine=aff)
            nib.save(niftiImage,savename)
            savename = saveDir + basename + '/SINGLE/labelsTs/case_' + str(ntrain_id) + '_' + str(nseg) + '.nii.gz'
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
    config_prep["folder_with_raw_data"] = saveDir + basename + '/MULTIPLE/'
    config_prep["preprocessing_output_dir"] = saveDir + basename + '/MULTIPLE/preprocessed'
    config_prep["folder_with_cropped_data"] = saveDir + basename + '/MULTIPLE/cropped/'

    f = open(saveDir + basename + '/MULTIPLE/configs/config_prep.json','w')
    json.dump(config_prep,f,indent=4)
    f.close()

    config_prep["folder_with_raw_data"] = saveDir + basename + '/SINGLE/'
    config_prep["preprocessing_output_dir"] = saveDir + basename + '/SINGLE/preprocessed'
    config_prep["folder_with_cropped_data"] = saveDir + basename + '/SINGLE/cropped/'

    f = open(saveDir + basename + '/SINGLE/configs/config_prep.json','w')
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
        config_train["output_folder"] = saveDir + basename + '/MULTIPLE/MODELS/'
        config_train["log_file"] = saveDir + basename + '/MULTIPLE/MODELS/log' + str(fold) + '.txt'
        if dimensions[basename] == '2d':
            config_train["plans_file_path"] = saveDir + basename + '/MULTIPLE/preprocessed/nnUNetPlansv2.1_plans_2D.pkl'
            config_train["folder_with_preprocessed_data"] = saveDir + basename + '/MULTIPLE/preprocessed/nnUNetData_plans_v2.1_2D_stage0'
        else:
            config_train["plans_file_path"] = saveDir + basename + '/MULTIPLE/preprocessed/nnUNetPlansv2.1_plans_3D.pkl'
            config_train["folder_with_preprocessed_data"] = saveDir + basename + '/MULTIPLE/preprocessed/nnUNetData_plans_v2.1_stage0'
            
        f = open(saveDir + basename + '/MULTIPLE/configs/config_train_' + str(fold) + '.json','w')
        json.dump(config_train,f,indent=4)
        f.close()
        
        config_train["output_folder"] = saveDir + basename + '/SINGLE/MODELS/'
        config_train["log_file"] = saveDir + basename + '/SINGLE/MODELS/log' + str(fold) + '.txt'
        if dimensions[basename] == '2d':
            config_train["plans_file_path"] = saveDir + basename + '/SINGLE/preprocessed/nnUNetPlansv2.1_plans_2D.pkl'
            config_train["folder_with_preprocessed_data"] = saveDir + basename + '/SINGLE/preprocessed/nnUNetData_plans_v2.1_2D_stage0'
        else:
            config_train["plans_file_path"] = saveDir + basename + '/SINGLE/preprocessed/nnUNetPlansv2.1_plans_3D.pkl'
            config_train["folder_with_preprocessed_data"] = saveDir + basename + '/SINGLE/preprocessed/nnUNetData_plans_v2.1_stage0'
            
        f = open(saveDir + basename + '/SINGLE/configs/config_train_' + str(fold) + '.json','w')
        json.dump(config_train,f,indent=4)
        f.close()
    """    

