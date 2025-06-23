
import math
import os
import json
import nibabel as nib
import numpy as np
import sys
import argparse
import pickle
from time import time

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

#   local imports
# sys.path.insert(0, "/net/people/plgrid/plgztabor/encodersUnet/")


# from augmentation.generators import prepare_generators
# from training.utils import *
# from training.loss import *
# from training.generic_UNet import *


from c_unet.augmentation.generators import prepare_generators

from c_unet.training.utils import *
from c_unet.training.loss import *
from c_unet.training.Probabilistic_UNet import *
from c_unet.training.Hierarchical_Prob_UNet import *

def run_online_evaluation(output, target):

    if isinstance(target,list):
        target = target[0]
        output = output[0]

    with torch.no_grad():

        num_classes = output.shape[1]
        output_softmax = softmax_helper(output)
        output_seg = output_softmax.argmax(1)
        target = target[:, 0]
        axes = tuple(range(1, len(target.shape)))
        tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
            fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
            fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

        return (2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)


def save_checkpoint(fname, network, optimizer, epoch,all_tr_losses, all_val_losses, best_epoch, best_loss):

    state_dict = network.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    print("saving checkpoint...",flush=True)

    save_this = {
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'plot_stuff': (all_tr_losses, all_val_losses),
        'best_stuff' : (best_epoch, best_loss)}

    torch.save(save_this, fname)
    print("saving done",flush=True)

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L    


if __name__ == '__main__':

    #torch.cuda.is_available = lambda: False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Used device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",required = True, help ="yml file with configuration")

    args = parser.parse_args()
    config_train = args.config

    with open(config_train, 'r') as config_file:
        configs = json.load(config_file)

    with open(configs['plans_file_path'], 'rb') as plans_file:
        plans = pickle.load(plans_file)

    if 'architecture_name' in configs.keys():
        architecture_name = configs['architecture_name']
    else:
        architecture_name = "Generic_UNet"
        
    print("using architecture",architecture_name)
    


    ######################################################################
    #######             prepare model                              #######
    ######################################################################

    network_type = configs['network_type']
    assert network_type in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], "Incorrect network type!"

    if '3d' in network_type:
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
    else:
        conv_op = nn.Conv2d
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

    possible_stages = list(plans['plans_per_stage'].keys())
    stage = possible_stages[-1]
    stage_plans = plans['plans_per_stage'][stage]
    net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
    net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']
    base_num_features = plans['base_num_features']
    num_input_channels = plans['num_modalities']
    num_classes = plans['num_classes'] + 1  # background is not in num_classes
    if 'conv_per_stage' in plans.keys():
        conv_per_stage = plans['conv_per_stage']
    else:
        conv_per_stage = 2

    if (network_type == '2d' or len(possible_stages) > 1) and not network_type == '3d_lowres':
        batch_dice = True
    else:
        batch_dice = False


    if architecture_name == "Probabilistic_UNet":

        net = Probabilistic_UNet(num_input_channels, base_num_features, num_classes,
                                 len(net_num_pool_op_kernel_sizes),
                                 conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                 dropout_op_kwargs,
                                 net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                 net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    elif architecture_name == "Probabilistic_UNet_v1":

        net = Probabilistic_UNet_v1(num_input_channels, base_num_features, num_classes,
                                 len(net_num_pool_op_kernel_sizes),
                                 conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                 dropout_op_kwargs,
                                 net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                 net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
        
    elif architecture_name == "Hierarchical_Prob_UNet":

        net = Hierarchical_Prob_UNet(num_input_channels, base_num_features, num_classes,
                             len(net_num_pool_op_kernel_sizes),
                             conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                             dropout_op_kwargs,
                             net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                             net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    else:
        print("undefined network type")
        exit()

    net.to(device)

    ########################################################################
    ##########                    PREPARE OPTIMIZER           ##############
    ########################################################################

    initial_lr = configs['initial_lr']
    momentum = configs['momentum']
    nesterov = configs['nesterov']
    weight_decay = 3e-5
    optimizer = torch.optim.SGD(net.parameters(), initial_lr, weight_decay=weight_decay,
                                             momentum=momentum, nesterov=nesterov)

    #################################################
    ####          lr scheduler                  #####
    #################################################
    lr_config = {}
    lr_config['factor'] = 0.2
    lr_config['mode'] = 'min'
    lr_config['patience'] = 20
    lr_config['optimizer'] = optimizer

    lr_scheduler = ReduceLROnPlateau(**lr_config)

    #################################################
    ######           loss                     #######
    #################################################

    beta_start = 0
    beta_stop = 1
    max_num_iterations = 300000
    beta_cycles = 10
    beta_ratio = 0.5
    beta_magnitude = 10

    beta_schedule = beta_magnitude* frange_cycle_cosine(beta_start, beta_stop, max_num_iterations, beta_cycles, beta_ratio)
    reconstruction_loss = nn.CrossEntropyLoss()

    reconstruction_loss = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

    #######################################################################
    ####                   PREPARE GENERATORS      ########################
    #######################################################################

    deep_supervision_scales = None
    weights = None

    tr_gen, val_gen = prepare_generators(config_train,deep_supervision_scales)

    #################################################
    ###                train config              ####
    #################################################
    numOfEpochs = configs["numOfEpochs"]
    tr_batches_per_epoch = configs["tr_batches_per_epoch"]
    val_batches_per_epoch = configs["val_batches_per_epoch"]
    checkpoint_frequency = configs["checkpoint_frequency"]
    outputDir = configs['output_folder']
    fold = configs['fold']
    log_file = configs['log_file']

    all_tr_losses = []
    all_val_losses = []
    all_dices = []

    startEpoch = 0

    bestValLoss = 1e30
    bestEpoch = 0

    if configs['continue_training']:
        print('loading model from',configs['checkpoint'],flush=True)
        fname = configs['checkpoint']
        saved_model = torch.load(fname, map_location=torch.device('cpu'))

        startEpoch = saved_model['epoch']
        all_tr_losses, all_val_losses = saved_model['plot_stuff']
        bestEpoch,bestValLoss = saved_model['best_stuff']

        optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        net.load_state_dict(saved_model['state_dict'])
        net.to(device)

        print('model loaded',flush=True)

    print('Config done')

    #################################################
    ###           training loop                  ####
    #################################################
    num_iterations = 1

    for epoch in range(startEpoch,numOfEpochs):

        #        if not isinstance(lr_scheduler, ReduceLROnPlateau):
        #            lr_scheduler.step()

        lr = poly_lr(epoch, numOfEpochs, initial_lr, 0.9)
        optimizer.param_groups[0]['lr'] = lr

        epoch_start_time = time()

        net.train()
        train_losses_epoch = []
        for batchNo in range(tr_batches_per_epoch):

            data_dict = next(tr_gen)
            
            data = data_dict['data']
            target = data_dict['target']

            data = data.to(device)
            target = target.to(device)
            
            output, kl_div = net.forward(data, target)
            l1 = reconstruction_loss(output, target)
            loss = l1 + beta_schedule[num_iterations] * kl_div.sum()

            train_losses_epoch.append(l1.item())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

            num_iterations += 1
            if num_iterations >= max_num_iterations:
                num_iterations = 1

        all_tr_losses.append(np.mean(train_losses_epoch))

        with torch.no_grad():
            net.eval()
            val_losses_epoch = []
            dices = []
            for _ in range(val_batches_per_epoch):

                data_dict = next(val_gen)
                
                data = data_dict['data']
                target = data_dict['target']
        
                data = data.to(device)
                target = target.to(device)

                output, kl_div = net.forward(data,target)
                l1 = reconstruction_loss(output, target)
                loss = l1 + beta_schedule[num_iterations] * kl_div.sum()
                
                val_losses_epoch.append(l1.item())
                dices.append(run_online_evaluation(output, target))

            all_val_losses.append(np.mean(val_losses_epoch))
            all_dices.append(np.mean(dices))

        epoch_end_time = time()

        print("epoch: ",epoch,", training loss: ", all_tr_losses[-1], ", validation loss: ", all_val_losses[-1],', validation dice: ',all_dices[-1],', this epoch took: ',epoch_end_time-epoch_start_time, 's',flush=True)

        
        f = open(log_file,'a')
        print("epoch: ",epoch,", training loss: ", all_tr_losses[-1], ", validation loss: ", all_val_losses[-1],', validation dice: ',all_dices[-1],', this epoch took: ',epoch_end_time-epoch_start_time, 's',file=f)
        f.close()

        if all_val_losses[-1] < bestValLoss:
            bestValLoss = all_val_losses[-1]
            bestEpoch = epoch
            fname = outputDir + '/fold_' + str(fold) + '_model_best.model'
            save_checkpoint(fname, net, optimizer, epoch,all_tr_losses, all_val_losses, bestEpoch, bestValLoss)

        if epoch%checkpoint_frequency == checkpoint_frequency-1:
            fname = outputDir + '/fold_' + str(fold) + '_model_latest.model'
            save_checkpoint(fname, net, optimizer, epoch,all_tr_losses, all_val_losses, bestEpoch, bestValLoss)

    fname = outputDir + '/fold_' + str(fold) + '_model_final.model'
    save_checkpoint(fname, net, optimizer, epoch,all_tr_losses, all_val_losses, bestEpoch, bestValLoss)

