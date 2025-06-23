import numpy as np
import nibabel as nib
import glob
import os

data_dir = '/U-Net/data/'
predictions_dir = '/U-Net/predictions/'

types = ['SINGLE', 'MULTIPLE']
datasets = ['BRAIN_TUMOR', 'KIDNEY', 'KNEE', 'PROSTATE', 'BRAIN_GROWTH', 'PANCREAS', 'SIJ', 'LUNG', 'HEART', 'PANCREATIC_LESION']
models = ['ensemble', 'bayesian', 'hier_probabilistic', 'probabilistic']


types = ['SINGLE', 'MULTIPLE', 'UNCERTAINTY_CLASS']
datasets = ['BRAIN_TUMOR_TASK1', 'PROSTATE_TASK1', 'PROSTATE_TASK2']
models = ['ensemble', 'bayesian', 'hier_probabilistic', 'probabilistic']


folds = ['fold0','fold1','fold2','fold3','fold4']

classes = {
    'BRAIN_TUMOR':3, 
    'BRAIN_TUMOR_TASK1':1,
    'KIDNEY':1, 'KNEE':1, 
    'PROSTATE':2, 
    'PROSTATE_TASK1':1,
    'PROSTATE_TASK2':1,
    'BRAIN_GROWTH':1, 
    'PANCREAS':1, 
    'SIJ':1, 
    'LUNG':1, 
    'HEART':2, 
    'PANCREATIC_LESION':1
}

for dataset in datasets:
    for typ in types:
        for model in models:

            ref_data_dir = os.path.join(data_dir,dataset,typ,'labelsTs')

            eval_data_dirs = []
            for fold in folds:
                eval_data_dirs.append(os.path.join(predictions_dir,dataset,typ,model,fold))

            fnames = sorted([os.path.basename(f) for f in glob.glob(ref_data_dir + '/*.nii.gz')])
            cases = list(set(['_'.join(f.split('_')[0:2]) for f in fnames]))

            results = np.zeros((len(cases),classes[dataset]),dtype=np.float32)
            for ncase,case in enumerate(cases):

                refNames = glob.glob(ref_data_dir + '/' + case + '_*.nii.gz')
                refs = []
                for refName in refNames:
                    ref = nib.load(refName).get_fdata()
                    refs.append(ref)

                evals = []
                for item in eval_data_dirs:
                    evNames = glob.glob(item + '/' + case + '_*.nii.gz')
                    for evName in evNames:
                        ev = nib.load(evName).get_fdata()
                        evals.append(ev)


                for label in range(1,classes[dataset]+1):
                    dum1 = [np.copy(item) for item in refs]
                    dum2 = [np.copy(item) for item in evals]
                    for i in range(len(dum1)):
                        dum1[i][dum1[i]!=label] = 0
                        dum1[i][dum1[i]!=0] = 1
                    for i in range(len(dum2)):
                        dum2[i][dum2[i]!=label] = 0
                        dum2[i][dum2[i]!=0] = 1
                    dum1 = np.asarray(dum1,dtype=np.float32)
                    dum2 = np.asarray(dum2,dtype=np.float32)
                    dum1 = np.mean(dum1,axis=0)
                    dum2 = np.mean(dum2,axis=0)
                    
                    dc = 2*np.sum(np.sqrt(dum1*dum2))/(np.sum(dum1) + np.sum(dum2) + 1e-8)
                    
                    results[ncase,label-1] = dc

            out_file_name = f'../stats/GDC/GDC_Ts_{dataset}_{typ}_{model}.txt'
            f = open(out_file_name,'w')
            for i in range(results.shape[0]):
                for j in range(results.shape[1]):
                    print(results[i,j],end=' ',file=f)
                print(file=f)
            f.close()
            #f = open('./stats/GDC/GDC_Ts.txt','a')
            #print(dataset,typ,model,np.mean(results,axis=0),np.std(results,axis=0),file=f)
            #f.close()

            #break
        #break
    #break
