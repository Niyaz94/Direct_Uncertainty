
from copy import deepcopy
from torch import nn
import torch
import numpy as np
from .neural_network import SegmentationNetwork
import torch.nn.functional
from torch import nn
import torch.nn.functional as F
from .utils import *


from torch.distributions import Normal, Independent, kl

from .based_prob_model import *

class Probabilistic_UNet(SegmentationNetwork):
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

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        """
        super(Probabilistic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

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

        self.prior_final_nonlin = nn.Softmax(dim=1)

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

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

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        self.latent_dim = 6
        self.prior_net = []
        self.prior_pooling = []

        self.posterior_net = []
        self.posterior_pooling = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            if d!=0:
                posterior_input_features = input_features
            else:
                posterior_input_features = input_channels + num_classes

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions to unet
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))

            # add convolutions to prior net
            self.prior_net.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.prior_pooling.append(pool_op(pool_op_kernel_sizes[d]))

            # add convolutions to posterior net
            self.posterior_net.append(StackedConvLayers(posterior_input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.posterior_pooling.append(pool_op(pool_op_kernel_sizes[d]))

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
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]

        # add final convolutions to unet
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # add final convolutions to prior net
        self.prior_net.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        self.prior_net.append(conv_op(final_num_features, 2 * self.latent_dim, 1, stride=1))

        # add final convolutions to posterior net
        self.posterior_net.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        self.posterior_net.append(conv_op(final_num_features, 2 * self.latent_dim, 1, stride=1))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
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
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        self.prior_net = nn.ModuleList(self.prior_net)
        self.prior_pooling = nn.ModuleList(self.prior_pooling)

        self.posterior_net = nn.ModuleList(self.posterior_net)
        self.posterior_pooling = nn.ModuleList(self.posterior_pooling)

        ######################################################
        #          feature combination                    ####
        ######################################################
        
        f_maps = 32
        no_convs_fcomb = 4
        fcomb_layers = []
        fcomb_layers.append(conv_op(final_num_features + self.latent_dim, f_maps, kernel_size=1))
        fcomb_layers.append(nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb-2):
            fcomb_layers.append(conv_op(f_maps, f_maps, kernel_size=1))
            fcomb_layers.append(nn.ReLU(inplace=True))

        self.fcomb_layers = nn.Sequential(*fcomb_layers)
        self.last_fcomb_layer = conv_op(f_maps, num_classes, kernel_size=1)

        ######################################################
        ######################################################

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        if torch.cuda.is_available():
            order_index = order_index.cuda()
        return torch.index_select(a, dim, order_index)

    def fcomb(self, feature_map, z):
        """
        Z is batch_size x latent_dim and feature_map is batch_size x no_channelsxDxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxWxD. Behavior is exactly the same as tf.tile (verified)
        """
        z = torch.unsqueeze(z,2)
        z = self.tile(z, 2, feature_map.shape[2])
        z = torch.unsqueeze(z,3)
        z = self.tile(z, 3, feature_map.shape[3])
        if len(feature_map.shape) > 4:
            z = torch.unsqueeze(z,4)
            z = self.tile(z, 4, feature_map.shape[4])
        
        #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
        feature_map = torch.cat((feature_map, z), dim=1)

        output = self.fcomb_layers(feature_map)
        return self.last_fcomb_layer(output)
            

    def forward(self, inp, target=None):

        ###########################################################
        ###                 Unet forward                        ###   
        ###########################################################

        x = inp

        skips = []
        #seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            #seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        unet_features = x

        #print('output of unet part',unet_features.shape,unet_features.sum())

        ###########################################################
        ##                    prior forward                      ##
        ###########################################################

        x = inp

        for d in range(len(self.prior_net) - 1):
            x = self.prior_net[d](x)
            if not self.convolutional_pooling:
                x = self.prior_pooling[d](x)

        print('output of unet part',unet_features.shape,unet_features.sum())

        encoding = x
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        if len(inp.shape) > 4:
            encoding = torch.mean(encoding, dim=4, keepdim=True)

        mu_log_sigma = self.prior_net[-1](encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        if len(inp.shape) > 4:
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        mu = torch.clamp(mu,min=1e-6,max=1e3)
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        log_sigma = torch.clamp(mu,min=-20,max=20)
        dist_prior = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)

        z_prior = dist_prior.sample()

        #print('prior latent shape',dist_prior)
        #print('prior latent sample shape',z_prior.shape)

        fcomb_out_prior = self.prior_final_nonlin(self.fcomb(unet_features,z_prior))

        ###########################################################
        ##                posterior forward                      ##
        ###########################################################

        if target != None:
            posterior_input = torch.cat ([inp,] + [target==i for i in range(self.num_classes)],axis = 1).float()
            x = posterior_input
            for d in range(len(self.posterior_net) - 1):
                x = self.posterior_net[d](x)
                if not self.convolutional_pooling:
                    x = self.posterior_pooling[d](x)

            encoding = x
            encoding = torch.mean(encoding, dim=2, keepdim=True)
            encoding = torch.mean(encoding, dim=3, keepdim=True)
            if len(posterior_input.shape) > 4:
                encoding = torch.mean(encoding, dim=4, keepdim=True)

            mu_log_sigma = self.posterior_net[-1](encoding)
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
            if len(posterior_input.shape) > 4:
                mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

            mu = mu_log_sigma[:,:self.latent_dim]
            mu = torch.clamp(mu,min=1e-6,max=1e3)
            log_sigma = mu_log_sigma[:,self.latent_dim:]
            log_sigma = torch.clamp(mu,min=-20,max=20)
            dist_posterior = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)

            kl_div = torch.sum(kl.kl_divergence(dist_posterior, dist_prior))

            z_posterior = dist_posterior.rsample()

            #print('posterior latent shape',dist_posterior)
            #print('posterior latent sample shape',z_posterior.shape)

            fcomb_out_posterior = self.fcomb(unet_features,z_posterior)

        ###########################################################
        if target != None:
            print('output shape',fcomb_out_posterior.shape)
            return fcomb_out_posterior,kl_div
        else:
            return fcomb_out_prior #,torch.Tensor([0.])

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp

    ###########################################################################################


class Probabilistic_UNet_v1(SegmentationNetwork):
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
    MAX_FILTERS_2D = 4800

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        """
        super(Probabilistic_UNet_v1, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

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

        self.prior_final_nonlin = nn.Softmax(dim = 1)

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

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

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        self.latent_dim = 1
        self.prior_net = []
        self.prior_pooling = []

        self.posterior_net = []
        self.posterior_pooling = []

        output_features = base_num_features
        input_features = input_channels

        posterior_input_features = input_channels + num_classes
        posterior_output_features = base_num_features

        prior_input_features = input_channels
        prior_output_features = base_num_features

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions to unet
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))

            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

            if d < num_pool-1:
                # add convolutions to prior net
                self.prior_net.append(StackedConvLayers(prior_input_features, prior_output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=ConvNonlin))
                if not self.convolutional_pooling:
                    self.prior_pooling.append(pool_op(pool_op_kernel_sizes[d]))

                prior_input_features = prior_output_features
                prior_output_features = int(np.round(prior_output_features * feat_map_mul_on_downscale))

                # add convolutions to posterior net
                self.posterior_net.append(StackedConvLayers(posterior_input_features, posterior_output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=ConvNonlin))
                if not self.convolutional_pooling:
                    self.posterior_pooling.append(pool_op(pool_op_kernel_sizes[d]))

                posterior_input_features = posterior_output_features
                posterior_output_features = int(np.round(posterior_output_features * feat_map_mul_on_downscale))



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
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]

        # add final convolutions to unet
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # add final convolutions to prior net
        self.prior_net.append(conv_op(prior_input_features, 2 * self.latent_dim, 1, stride=1))

        # add final convolutions to posterior net
        self.posterior_net.append(conv_op(posterior_input_features, 2 * self.latent_dim, 1, stride=1))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
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
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        self.prior_net = nn.ModuleList(self.prior_net)
        self.prior_pooling = nn.ModuleList(self.prior_pooling)

        self.posterior_net = nn.ModuleList(self.posterior_net)
        self.posterior_pooling = nn.ModuleList(self.posterior_pooling)

        ######################################################
        #          feature combination                    ####
        ######################################################

        f_maps = 32
        no_convs_fcomb = 4
        fcomb_layers = []
        fcomb_layers.append(conv_op(final_num_features + self.latent_dim, f_maps, kernel_size=1))
        fcomb_layers.append(nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb-2):
            fcomb_layers.append(conv_op(f_maps, f_maps, kernel_size=1))
            fcomb_layers.append(nn.ReLU(inplace=True))

        self.fcomb_layers = nn.Sequential(*fcomb_layers)
        self.last_fcomb_layer = conv_op(f_maps, num_classes, kernel_size=1)

        ######################################################
        ######################################################

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        if torch.cuda.is_available():
            order_index = order_index.cuda()
        return torch.index_select(a, dim, order_index)

    def fcomb(self, feature_map, z):
        """
        Z is batch_size x latent_dim and feature_map is batch_size x no_channelsxDxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxWxD. Behavior is exactly the same as tf.tile (verified)
        """
        z = torch.unsqueeze(z,2)
        z = self.tile(z, 2, feature_map.shape[2])
        z = torch.unsqueeze(z,3)
        z = self.tile(z, 3, feature_map.shape[3])
        if len(feature_map.shape) > 4:
            z = torch.unsqueeze(z,4)
            z = self.tile(z, 4, feature_map.shape[4])

        #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
        feature_map = torch.cat((feature_map, z), dim=1)

        output = self.fcomb_layers(feature_map)
        return self.last_fcomb_layer(output)


    def forward(self, inp, target=None):

        ###########################################################
        ###                 Unet forward                        ###
        ###########################################################

        x = inp

        skips = []
        #seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            #seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        unet_features = x

        #print('output of unet part',unet_features.shape,unet_features.sum())

        ###########################################################
        ##                    prior forward                      ##
        ###########################################################

        x = inp

        for d in range(len(self.prior_net) - 1):
            x = self.prior_net[d](x)
            if not self.convolutional_pooling:
                x = self.prior_pooling[d](x)

        print('output of unet part',unet_features.shape,unet_features.sum())

        encoding = x
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        if len(inp.shape) > 4:
            encoding = torch.mean(encoding, dim=4, keepdim=True)

        mu_log_sigma = self.prior_net[-1](encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        if len(inp.shape) > 4:
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        mu = torch.clamp(mu,min=1e-6,max=1e3)
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        log_sigma = torch.clamp(mu,min=-20,max=20)
        dist_prior = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)

        z_prior = dist_prior.sample()

        #print('prior latent shape',dist_prior)
        #print('prior latent sample shape',z_prior.shape)

        fcomb_out_prior = self.prior_final_nonlin(self.fcomb(unet_features,z_prior))

        ###########################################################
        ##                posterior forward                      ##
        ###########################################################

        if target != None:
            posterior_input = torch.cat ([inp,] + [target==i for i in range(self.num_classes)],axis = 1).float()
            x = posterior_input
            for d in range(len(self.posterior_net) - 1):
                x = self.posterior_net[d](x)
                if not self.convolutional_pooling:
                    x = self.posterior_pooling[d](x)

            encoding = x
            encoding = torch.mean(encoding, dim=2, keepdim=True)
            encoding = torch.mean(encoding, dim=3, keepdim=True)
            if len(posterior_input.shape) > 4:
                encoding = torch.mean(encoding, dim=4, keepdim=True)

            mu_log_sigma = self.posterior_net[-1](encoding)
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
            if len(posterior_input.shape) > 4:
                mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

            mu = mu_log_sigma[:,:self.latent_dim]
            mu = torch.clamp(mu,min=1e-6,max=1e3)
            log_sigma = mu_log_sigma[:,self.latent_dim:]
            log_sigma = torch.clamp(mu,min=-20,max=20)
            dist_posterior = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)

            kl_div = torch.sum(kl.kl_divergence(dist_posterior, dist_prior))

            z_posterior = dist_posterior.rsample()

            #print('posterior latent shape',dist_posterior)
            #print('posterior latent sample shape',z_posterior.shape)

            fcomb_out_posterior = self.fcomb(unet_features,z_posterior)

        ###########################################################
        if target != None:
            print('output shape',fcomb_out_posterior.shape)
            return fcomb_out_posterior,kl_div
        else:
            return fcomb_out_prior #,torch.Tensor([0.])

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp

