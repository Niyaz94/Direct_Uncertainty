import glob
import shutil
import os
import nibabel as nib
import numpy as np
import json

from skimage.morphology import closing
from skimage.morphology import square
from scipy.stats import wilcoxon

def dice(reference,evaluated,labels):

    dices = []
    for label in labels[1:]:
        dum1 = np.copy(reference)
        dum2 = np.copy(evaluated)
        dum1[dum1!=label] = 0
        dum1[dum1!=0] = 1
        dum2[dum2!=label] = 0
        dum2[dum2!=0] = 1
        #CImg((dum1 + dum2)[:,:,0]).display(case + ' ' + str(label));
        dc = 2*np.sum(dum1*dum2)/(np.sum(dum1) + np.sum(dum2) + 1e-8)
        dices.append(dc)

    return np.array(dices,dtype=np.float32)


data_dir = '/U-Net/data/'
pred_dir = '/U-Net/predictions/'
save_dir = '/U-Net/stats/UNCERTAINTY_CLASS/'

#classes = {'KIDNEY':1, 'KNEE':1, 'PROSTATE':2, 'BRAIN_GROWTH':1, 'PANCREAS':1, 'SIJ':1, 'LUNG':1, 'HEART':2, 'PANCREATIC_LESION':1}

classes = {'PROSTATE_TASK1':1, 'PROSTATE_TASK2':1, 'BRAIN_TUMOR_TASK1':1}


datasets = classes.keys()

model = '/UNCERTAINTY_CLASS/'

TH = 1e-4

for dataset in datasets:

    # read from [~/data/__DATASET__/UNCERTAINTY_CLASS] directory
    rnames = glob.glob(data_dir + dataset + model + 'labelsTs/*')
    cases = [os.path.basename(f) for f in rnames]
    # read from [~/data/__DATASET__/UNCERTAINTY_CLASS] directory
    data_dict_fname = data_dir + dataset + model + 'dataset.json'
    with open(data_dict_fname,'r') as f:
        data_dict = json.load(f)

    labels = sorted([int(i) for i in data_dict["labels"].keys()])
    print(labels)

    results1 = np.zeros((len(cases),len(labels)-1), dtype = np.float32)
    results2 = np.zeros((len(cases),len(labels)-1), dtype = np.float32)
    results3 = np.zeros((len(cases),len(labels)-1), dtype = np.float32)

    for ncase, case in enumerate(cases):

        # read from [~/predictions/__DATASET__/UNCERTAINTY_CLASS] directory
        # if not os.path.isfile(pred_dir + dataset + model + 'ensemble/' + case):
        #     print(dataset,case)

        # read from [~/data/__DATASET__/UNCERTAINTY_CLASS] directory
        gt_im = nib.load(data_dir + dataset + model + 'labelsTs/' + case).get_fdata()
        
        # read from [~/predictions/__DATASET__/UNCERTAINTY_CLASS] directory
        ev_im = nib.load(pred_dir + dataset + model + 'ensemble/' + case).get_fdata()
        
        # one label less than the number of labels
        # read from [~/predictions/__DATASET__/SINGLE] directory
        single_ims = [nib.load(pred_dir + dataset + '/SINGLE/ensemble/fold' + str(i) + '/' + case).get_fdata() for i in range(5)]
        
        single_ims = np.mean(np.asarray(single_ims,dtype=np.float32),axis = 0)
        mean_labels = np.unique(single_ims) # incase of HEART dataset, mean_labels max is 2; in [labels] max is 3
        new_labels = [i for i in mean_labels if np.min([abs(i-l) for l in labels]) > TH]
        # [0, 1, 2, 3] -[2]-> [2, 1, 0, 1]
        for i in new_labels:
            single_ims[single_ims==i] = np.max(labels)

        # read from [~/predictions/__DATASET__/MULTIPLE/] directory
        multiple_ims = [nib.load(pred_dir + dataset + '/MULTIPLE/ensemble/fold' + str(i) + '/' + case).get_fdata() for i in range(5)]
        multiple_ims = np.mean(np.asarray(multiple_ims,dtype=np.float32),axis = 0)
        mean_labels = np.unique(multiple_ims)
        new_labels = [i for i in mean_labels if np.min([abs(i-l) for l in labels]) > TH]
        for i in new_labels:
            multiple_ims[multiple_ims==i] = np.max(labels)

        r1 = dice(gt_im,ev_im,labels)
        r2 = dice(gt_im,single_ims,labels)
        r3 = dice(gt_im,multiple_ims,labels)

        results1[ncase] = np.copy(r1)
        results2[ncase] = np.copy(r2)
        results3[ncase] = np.copy(r3)

    fname = save_dir + dataset + '_direct.txt'
    f = open(fname,'w')
    for i in range(len(cases)):
        for j in range(len(labels)-1):
            print(results1[i,j],end = ' ', file = f)
        print(file=f)
    f.close()

    fname = save_dir + dataset + '_single.txt'
    f = open(fname,'w')
    for i in range(len(cases)):
        for j in range(len(labels)-1):
            print(results2[i,j],end = ' ', file = f)
        print(file=f)
    f.close()

    fname = save_dir + dataset + '_multiple.txt'
    f = open(fname,'w')
    for i in range(len(cases)):
        for j in range(len(labels)-1):
            print(results3[i,j],end = ' ', file = f)
        print(file=f)
    f.close()

    fname = save_dir + 'medians.txt'
    f = open(fname,'a')
    print('##############################',file=f)
    print(dataset, np.median(results1,axis=0), np.median(results2,axis=0), np.median(results3,axis=0),file=f)
    print(dataset,np.quantile(results1,(0.025,0.975),axis=0), np.quantile(results2,(0.025,0.975),axis=0), np.quantile(results3,(0.025,0.975),axis=0),file=f)
    f.close()

    fname = save_dir + 'stat_test.txt'
    f = open(fname,'a')
    print('########################',file=f)
    for i in range(results1.shape[1]):
        print(dataset, wilcoxon(results1[:,i],results2[:,i]),wilcoxon(results1[:,i],results3[:,i]),wilcoxon(results3[:,i],results2[:,i]),file=f)
    f.close()
    


