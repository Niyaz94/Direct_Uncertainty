import numpy as np
import nibabel as nib
import glob
import os

data_dir = '/U-Net/data/'
predictions_dir = '/U-Net/predictions/'

types = ['SINGLE', 'MULTIPLE']
datasets = ['BRAIN_TUMOR', 'KIDNEY', 'KNEE', 'PROSTATE', 'BRAIN_GROWTH', 'PANCREAS', 'SIJ', 'LUNG', 'HEART', 'PANCREATIC_LESION']
models = ['ensemble', 'bayesian', 'hier_probabilistic', 'probabilistic']

types = ['SINGLE', 'MULTIPLE']
datasets = ['KIDNEY', 'PANCREAS', 'PANCREATIC_LESION']
models = ['ensemble']

folds = ['fold0','fold1','fold2','fold3','fold4']

classes = {
    'BRAIN_TUMOR':3, 'KIDNEY':1, 'KNEE':1, 'PROSTATE':2, 'BRAIN_GROWTH':1, 
    'PANCREAS':1, 'SIJ':1, 'LUNG':1, 'HEART':2, 'PANCREATIC_LESION':1
}

def QDICE(pred, target, num_bins=10):
   
    bin_edges_low = [        i / num_bins for i in range(num_bins)]
    bin_edges_high = [edge + 1 / num_bins for edge in bin_edges_low]
    bin_edges_high[-1] += 1e-5  # small nudge to include 1.0

    dice_scores = []
    for low, high in zip(bin_edges_low, bin_edges_high):
        
        pred_band = ((pred >= low) & (pred < high)).astype(np.uint8)
        target_band = ((target >= low) & (target < high)).astype(np.uint8)

        intersection = np.sum(pred_band * target_band)
        union = np.sum(pred_band) + np.sum(target_band)

        if union > 0:
            dice = 2 * intersection / union
            dice_scores.append(dice)

    return np.mean(dice_scores) if dice_scores else 1.0

def DICE(pred, target, num_thresholds=10):
    thresholds = np.linspace(0, 1, num_thresholds + 2)[1:-1]  # avoid 0 and 1
    dice_scores = []

    for t in thresholds:
        pred_bin = (pred >= t).astype(np.uint8)
        target_bin = (target >= t).astype(np.uint8)
        
        # dc = 2*np.sum(np.sqrt(dum1*dum2))/(np.sum(dum1) + np.sum(dum2) + 1e-8)

        intersection = np.sum(pred_bin * target_bin)
        union = np.sum(pred_bin) + np.sum(target_bin)

        if union > 0:
            dice = 2 * intersection / union
            dice_scores.append(dice)

    return np.mean(dice_scores) if dice_scores else 1.0


N_LEVELS = 10

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
            results1 = np.zeros((len(cases),classes[dataset]),dtype=np.float32)

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

                    dc = QDICE(dum1,dum2,N_LEVELS) #len(ref)) # len(ref) should be used instead of N_LEVELS?
                    results[ncase,label-1] = dc

                    dc = DICE(dum1,dum2,N_LEVELS) #len(ref)) # len(ref) should be used instead of N_LEVELS?
                    results1[ncase,label-1] = dc
                    

            out_file_name = f'./stats/QDICE/QDICE_Ts_{dataset}_{typ}_{model}.txt'
            
            f = open(out_file_name,'w')
            for i in range(results.shape[0]):
                for j in range(results.shape[1]):
                    print(results[i,j],end=' ',file=f)
                print(file=f)
            f.close()
            f = open('./stats/QDICE/QDICE_Ts.txt','a')
            print(dataset,typ,model,np.mean(results,axis=0),np.std(results,axis=0),file=f)
            f.close()

            out_file_name = f'./stats/DICE/DICE_Ts_{dataset}_{typ}_{model}.txt'
            f = open(out_file_name,'w')
            for i in range(results1.shape[0]):
                for j in range(results1.shape[1]):
                    print(results1[i,j],end=' ',file=f)
                print(file=f)
            f.close()
            f = open('./stats/DICE/DICE_Ts.txt','a')
            print(dataset,typ,model,np.mean(results1,axis=0),np.std(results1,axis=0),file=f)
            f.close()

            #break
        #break
    #break
