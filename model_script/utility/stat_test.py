import numpy as np
import nibabel as nib
import glob
import os

import numpy as np
from scipy.stats import wilcoxon

import sys

METRIC = sys.argv[1]

datasets = ['BRAIN_TUMOR_TASK1', 'KIDNEY', 'KNEE', 'PROSTATE', 'PROSTATE_TASK1', 'PROSTATE_TASK2', 'BRAIN_GROWTH', 'PANCREAS', 'SIJ', 'LUNG', 'HEART', 'PANCREATIC_LESION']
models = ['ensemble', 'bayesian', 'hier_probabilistic', 'probabilistic']


classes = {'BRAIN_TUMOR_TASK1':1, 'KIDNEY':1, 'KNEE':1, 'PROSTATE':2, 'PROSTATE_TASK1':1, 'PROSTATE_TASK2':1, 'BRAIN_GROWTH':1, 'PANCREAS':1, 'SIJ':1, 'LUNG':1, 'HEART':2, 'PANCREATIC_LESION':1}

for dataset in datasets:
        for model in models:

            print(dataset,model,end=' ')

            out_file_name = f'/U-Net/stats/{METRIC}/{METRIC}_Ts_{dataset}_SINGLE_{model}.txt'
            f = open(out_file_name,'r')
            lines = f.readlines()
            f.close()

            results1 = [list(map(float,f.split())) for f in lines]
            results1 = np.asarray(results1,dtype=np.float32)

            out_file_name = f'/U-Net/stats/{METRIC}/{METRIC}_Ts_{dataset}_MULTIPLE_{model}.txt'
            f = open(out_file_name,'r')
            lines = f.readlines()
            f.close()

            results2 = [list(map(float,f.split())) for f in lines]
            results2 = np.asarray(results2,dtype=np.float32)


            f = open(f'/U-Net/stats/{METRIC}/{METRIC}_Ts_stats.txt','a')
            print(dataset,model,end=' ',file=f)
            print('SINGLE',np.median(results1,axis=0),np.quantile(results1,(0.025,0.975),axis=0),file=f)
            print(dataset,model,end=' ',file=f)
            print('MULTIPLE',np.median(results2,axis=0),np.quantile(results2,(0.025,0.975),axis=0),file=f)
            print(file=f)
            f.close()

            f = open(f'/U-Net/stats/{METRIC}/{METRIC}_Ts_Wilcoxon.txt','a')
            print(dataset,model,end=' ',file=f)
            for k in range(results1.shape[1]):
                p_value = 1.0
                try:
                    _, p_value = wilcoxon(results1[:,k],results2[:,k])
                except:
                    pass
                print(p_value,end=' ')
                print(p_value,end=' ',file=f)
            print(file=f)
            print()
            f.close()

            #break
        #break
