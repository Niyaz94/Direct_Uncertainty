from copy import deepcopy
from torch import nn
import torch
import numpy as np
from .neural_network import SegmentationNetwork
import torch.nn.functional
from torch import nn
import torch.nn.functional as F
from .utils import *
from .based_model import *

class Bayesian_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480
    # repa
    BAYESIAN_TYPE = "flip"

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(
        self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,feat_map_mul_on_downscale=2, 
        conv_op=nn.Conv2d,norm_op=nn.BatchNorm2d, norm_op_kwargs=None,dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
        final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,conv_kernel_sizes=None,
        upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,max_num_features=None, 
        basic_block=ConvDropoutNormNonlin,seg_output_use_bias=False
    ):
        
        super(Bayesian_UNet, self).__init__()
        
        
        self.convolutional_upsampling   = convolutional_upsampling
        self.convolutional_pooling      = convolutional_pooling
        self.upscale_logits             = upscale_logits
        
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin             = nonlin
        self.nonlin_kwargs      = nonlin_kwargs
        self.dropout_op_kwargs  = dropout_op_kwargs
        self.norm_op_kwargs     = norm_op_kwargs
        self.weightInitializer  = weightInitializer
        self.conv_op            = conv_op
        self.norm_op            = norm_op
        self.dropout_op         = dropout_op
        self.num_classes        = num_classes
        self.final_nonlin       = final_nonlin
        self._deep_supervision  = deep_supervision
        self.do_ds              = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))
        
        

        self.input_shape_must_be_divisible_by   = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes               = pool_op_kernel_sizes
        self.conv_kernel_sizes                  = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features
            
        self.conv_op    = choosen_layer(self.conv_op,self.BAYESIAN_TYPE,"norm")
        self.transpconv = choosen_layer(transpconv,self.BAYESIAN_TYPE,"bays")

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        # print(f"for d in range(num_pool) length is: {num_pool}")  -> 3
        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(
                input_features, output_features, num_conv_per_stage,self.conv_op, self.conv_kwargs, self.norm_op,
                self.norm_op_kwargs, self.dropout_op,self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                first_stride, basic_block=basic_block
            ))
            self.conv_op    = choosen_layer(self.conv_op,self.BAYESIAN_TYPE,"norm")
            
            
            
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
                
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels
            

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding']     = self.conv_pad_sizes[num_pool]
        
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(
                input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,self.norm_op, 
                self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,self.nonlin_kwargs, first_stride,
                basic_block=basic_block
            ),
            StackedConvLayers(
                output_features, 
                final_num_features, 1, 
                choosen_layer(self.conv_op,self.BAYESIAN_TYPE,"norm"),  
                self.conv_kwargs,self.norm_op, self.norm_op_kwargs, 
                self.dropout_op, self.dropout_op_kwargs, self.nonlin,self.nonlin_kwargs, basic_block=basic_block
            )
        ))        

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        # print(f"for u in range(num_pool) length is: {num_pool}")  -> 3
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(self.transpconv(
                    nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                    pool_op_kernel_sizes[-(u + 1)], bias=False
                ))
                # self.transpconv = choosen_layer(self.transpconv,self.BAYESIAN_TYPE)
                # self.transpconv = (self.transpconv)
                

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(
                    n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,self.conv_op, self.conv_kwargs, 
                    self.norm_op, self.norm_op_kwargs, self.dropout_op,self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,basic_block=basic_block
                ),
                StackedConvLayers(
                    nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
                )
            )

        self.conv_op = choosen_layer(self.conv_op,self.BAYESIAN_TYPE,"bays")
        # print(f"for ds in range(len(self.conv_blocks_localization)) length is: {len(self.conv_blocks_localization)}") -> 3
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(
                self.conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,1, 1, 0, 1, 1, seg_output_use_bias)
            )
            # self.conv_op = choosen_layer(self.conv_op,self.BAYESIAN_TYPE,"norm")
            

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(
                    Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),mode=upsample_mode)
                )
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization   = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context        = nn.ModuleList(self.conv_blocks_context)
        self.td                         = nn.ModuleList(self.td)
        self.tu                         = nn.ModuleList(self.tu)
        self.seg_outputs                = nn.ModuleList(self.seg_outputs)
        
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x, target = None):

        #print('self.convolutional_pooling',self.convolutional_pooling)
        #print("inside forward")
        #print("x shape:", x.shape)

        skips = []
        seg_outputs = []
        kl_ = []
        
        # for d in range(len(self.conv_blocks_context)):
            # print(f"Local layers: {str(self.conv_blocks_context[d])}")
        
        for d in range(len(self.conv_blocks_context) - 1):
            # x = self.conv_blocks_context[d](x)
            x ,kl= self.conv_blocks_context[d](x)
            kl_.append(kl)
            #print("x shape 1", x.shape)
            skips.append(x)
            if not self.convolutional_pooling:
                # x = self.td[d](x)
                x , kl= self.td[d](x)
                kl_.append(kl)
                #print("x shape 2", x.shape)
                
        # print(f"X len:{len(x)}")

        # x = self.conv_blocks_context[-1](x)
        x, kl = self.conv_blocks_context[-1][0](x)
        kl_.append(kl)
        x, kl = self.conv_blocks_context[-1][1](x)
        kl_.append(kl)
        
        
        # for d in range(len(self.tu)):
        #     print(f"Local layers: {str(self.conv_blocks_localization[d])}")
        
        # for u in range(len(self.tu)):
            # layer = self.seg_outputs[u]
            # 
            # Print parameters of the current layer
            # print(f"\nLayer {u}: {layer.__class__.__name__}")
            # for name, param in layer.named_parameters():
                # print(f"  {name}: {param.shape}")

        for u in range(len(self.tu)):
            # x = self.tu[u](x)
            if isinstance(x, tuple):
                x, kl = self.tu[u](x)
                kl_.append(kl)
            else:
                x = self.tu[u](x)
            
            
            # print(f"Input Type: {type(x)}, Skips Type: {type(skips[-(u + 1)])}")
            if isinstance(x, tuple):
                x, kl = x
                
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            # x, kl = torch.cat((x, skips[-(u + 1)]), dim=1)
            kl_.append(kl)
            
            #print("x shape 4", x.shape)
            # x = self.conv_blocks_localization[u](x)
            x, kl = self.conv_blocks_localization[u][0](x)
            kl_.append(kl)
            x, kl = self.conv_blocks_localization[u][1](x)
            kl_.append(kl)
            #print("x shape 5", x.shape)
            # print(self.seg_outputs[u],type(x),x.shape)
            # for name, param in self.seg_outputs[u].named_parameters():
            #     print(f"  {name}: {param.shape}")
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            
        kl = torch.mean(torch.stack(kl_), dim=0)
            

        if target != None:
            if isinstance(seg_outputs[-1], tuple):
                return seg_outputs[-1][0], kl
            else:
                return seg_outputs[-1], kl

        else:
            final_nonlin = nn.Softmax(dim = 1)
            if isinstance(seg_outputs[-1], tuple):
                return final_nonlin(seg_outputs[-1][0])
            else:
                return final_nonlin(seg_outputs[-1])

    def forward_old(self, x):

        #print('self.convolutional_pooling',self.convolutional_pooling)
        #print("inside forward")
        #print("x shape:", x.shape)

        skips = []
        seg_outputs = []
        kl_ = []
        
        # for d in range(len(self.conv_blocks_context)):
            # print(f"Local layers: {str(self.conv_blocks_context[d])}")
        
        for d in range(len(self.conv_blocks_context) - 1):
            # x = self.conv_blocks_context[d](x)
            x ,kl= self.conv_blocks_context[d](x)
            kl_.append(kl)
            #print("x shape 1", x.shape)
            skips.append(x)
            if not self.convolutional_pooling:
                # x = self.td[d](x)
                x , kl= self.td[d](x)
                kl_.append(kl)
                #print("x shape 2", x.shape)
                
        # print(f"X len:{len(x)}")

        # x = self.conv_blocks_context[-1](x)
        x, kl = self.conv_blocks_context[-1][0](x)
        kl_.append(kl)
        x, kl = self.conv_blocks_context[-1][1](x)
        kl_.append(kl)
        
        
        # for d in range(len(self.tu)):
        #     print(f"Local layers: {str(self.conv_blocks_localization[d])}")

        for u in range(len(self.tu)):
            # x = self.tu[u](x)
            if isinstance(x, tuple):
                x, kl = self.tu[u](x)
                kl_.append(kl)
            else:
                x = self.tu[u](x)
            
            #print("x shape 3", x.shape)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            # x, kl = torch.cat((x, skips[-(u + 1)]), dim=1)
            kl_.append(kl)
            #print("x shape 4", x.shape)
            # x = self.conv_blocks_localization[u](x)
            x, kl = self.conv_blocks_localization[u][0](x)
            kl_.append(kl)
            x, kl = self.conv_blocks_localization[u][1](x)
            kl_.append(kl)
            #print("x shape 5", x.shape)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            
        kl = torch.mean(torch.stack(kl_), dim=0)
        
        # if self.training:
            # Training-specific behavior
            # print("Forward pass in training mode")
            # change _deep_supervision to false
            # if self._deep_supervision and self.do_ds:
                # return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), kl
            # else:
                # return seg_outputs[-1], kl
        # else:
            # Testing-specific behavior
            # print("Forward pass in testing mode")
            # change _deep_supervision to false
            # if self._deep_supervision and self.do_ds:
                # return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
            # else:
                # return seg_outputs[-1]
            
        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), kl
        else:
            return seg_outputs[-1], kl