from torch import nn
import torch
from copy import deepcopy
from  bayesian_torch.layers import Conv2dReparameterization, Conv3dReparameterization, ConvTranspose2dReparameterization,ConvTranspose3dReparameterization, Conv2dFlipout,ConvTranspose2dFlipout,Conv3dFlipout,ConvTranspose3dFlipout



class ConvDropoutNormNonlin(nn.Module):
    def __init__(
        self, input_channels, output_channels,conv_op=nn.Conv2d, conv_kwargs=None,norm_op=nn.BatchNorm2d, 
        norm_op_kwargs=None,dropout_op=nn.Dropout2d, dropout_op_kwargs=None,nonlin=nn.LeakyReLU, nonlin_kwargs=None
    ):
        super(ConvDropoutNormNonlin, self).__init__()
        
        # print(conv_kwargs,"Hi..........")
        # print(str(conv_op))
        
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        
        layer_str_name=str(conv_op).split('.')[-1].rstrip("'>")
        # ConvTranspose3dReparameterization
        # if layer_str_name in ["Conv3dReparameterization"]:
                # conv_kwargs = {**conv_kwargs,"prior_mean": 0,"prior_variance": 1,"posterior_mu_init": 0,"posterior_rho_init": -3}
            
        # print(,conv_op.__class__)
        
        
        # print(conv_op,conv_kwargs)
        

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.iteration= []

        # (Tensor, Tensor, Tensor, bool, int, int, int)
        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        self.iteration.append(self.conv)
        
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
            self.iteration.append(self.dropout)
        else:
            self.dropout = None
            
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)
        self.iteration+=[self.instnorm,self.lrelu]
        
        
    def __iter__(self):
        return iter(self.iteration)


    def forward(self, x):
        
        # for element in x:
        #     print(f"ConvDropoutNormNonlin: {type(element)}, Size: {element.size()}, Name: {self.conv}")
        # print("------------------------------------------------------")
        
        if isinstance(layer, (Conv2dFlipout,ConvTranspose2dFlipout,Conv3dFlipout,ConvTranspose3dFlipout,Conv2dReparameterization, ConvTranspose2dReparameterization,Conv3dReparameterization,ConvTranspose3dReparameterization)):
        # if isinstance(self.conv, (Conv2dReparameterization, ConvTranspose2dReparameterization,Conv3dReparameterization,ConvTranspose3dReparameterization)):
            x, kl = self.conv(x)
        else:
            x = self.conv(x)
            kl = torch.tensor(0.0, device=x.device)
            
            
        if self.dropout is not None:
            x = self.dropout(x)
            
        return self.lrelu(self.instnorm(x)), kl
    
    def forward_ff(self, x):
        
        # for element in x:
        #     print(f"ConvDropoutNormNonlin: {type(element)}, Size: {element.size()}, Name: {self.conv}")
        # print("------------------------------------------------------")
        
            
        kl = torch.tensor(0.0, device=x[0].device)
        
        if isinstance(self.conv, (Conv2dReparameterization, ConvTranspose2dReparameterization,Conv3dReparameterization,ConvTranspose3dReparameterization)):
            x , kl= self.conv(x)
        else: 
            x = self.conv(x)
            
        if self.dropout is not None:
            x = self.dropout(x)
                
        return self.lrelu(self.instnorm(x)),kl

class StackedConvLayers(nn.Module):
    def __init__(
        self, input_feature_channels, output_feature_channels, num_convs,conv_op=nn.Conv2d, conv_kwargs=None,
        norm_op=nn.BatchNorm2d, norm_op_kwargs=None,dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin
    ):
        '''
            stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. 
            The other parameters affect all layers
        '''
        self.input_channels  = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(*(
            [basic_block(
                input_feature_channels, output_feature_channels, (self.conv_op),self.conv_kwargs_first_conv,self.norm_op, 
                self.norm_op_kwargs,self.dropout_op, self.dropout_op_kwargs,self.nonlin, self.nonlin_kwargs
            )] +
            [basic_block(
                output_feature_channels, output_feature_channels, (self.conv_op),self.conv_kwargs,self.norm_op, self.norm_op_kwargs, 
                self.dropout_op, self.dropout_op_kwargs,self.nonlin, self.nonlin_kwargs
            ) for _ in range(num_convs - 1)]            
        ))

    def forward_ff(self, x):
        
        # for element in x:
        #     print(f"Type: {type(element)}, Size: {element.size()}")
        # print("------------------------------------------------------")
            
        kl_sum = torch.tensor(0.0, device=x[0].device)
        for layer in self.blocks:
            if isinstance(layer, (Conv2dFlipout,ConvTranspose2dFlipout,Conv3dFlipout,ConvTranspose3dFlipout,Conv2dReparameterization, ConvTranspose2dReparameterization,Conv3dReparameterization,ConvTranspose3dReparameterization)):
                x,kl = layer(x)  
                kl_sum += kl
            else:
                x = layer(x)
        return x,kl_sum
    
    def forward(self, x):
        
        # kl_sum = torch.tensor(0.0, device=(x[0].device if isinstance(x, tuple) else x.device))
        kl_sum = [torch.tensor(0.0, device=(x[0].device if isinstance(x, tuple) else x.device))]
   
        for sub_block in self.blocks:
            for layer in sub_block:
                # print(f"Layer name is {str(layer)}")
                if isinstance(layer, (Conv2dFlipout,ConvTranspose2dFlipout,Conv3dFlipout,ConvTranspose3dFlipout,Conv2dReparameterization, ConvTranspose2dReparameterization,Conv3dReparameterization,ConvTranspose3dReparameterization)):
                    x,kl = layer(x)  
                    kl_sum.append(kl)
                else:
                    x = layer(x)
                
        
        kl_sum=sum(kl_sum)
        # /len(kl_sum)
        
        return x, kl_sum

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        
        # mode: bilinear (for 2D images), trilinear (for 3D images)
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(
            x, 
            size=self.size, 
            scale_factor=self.scale_factor, 
            mode=self.mode,
            align_corners=self.align_corners
        )

def choosen_layer (layer,b_type="repa",layer_type="bays"): 
    
    # return layer 
    
    def bayesian_type(b_shape,b_layer):
        if b_type=="repa" and b_shape=="2d" and b_layer=="norm":
            return Conv2dReparameterization
        elif b_type=="repa" and b_shape=="3d" and b_layer=="norm":
            return Conv3dReparameterization
        elif b_type=="flip" and b_shape=="2d" and b_layer=="norm":
            return Conv2dFlipout
        elif b_type=="flip" and b_shape=="3d" and b_layer=="norm":
            return Conv3dFlipout
        elif b_type=="repa" and b_shape=="2d" and b_layer=="tran":
            return ConvTranspose2dReparameterization
        elif b_type=="repa" and b_shape=="3d" and b_layer=="tran":
            return ConvTranspose3dReparameterization
        elif b_type=="flip" and b_shape=="2d" and b_layer=="tran":
            return ConvTranspose2dFlipout
        elif b_type=="flip" and b_shape=="3d" and b_layer=="tran":
            return ConvTranspose3dFlipout
            
    # print(f"Layer name is: {str(layer)} , {layer in [Conv3dFlipout,Conv3dReparameterization]}")
    if layer == nn.Conv2d:
        return layer if layer_type=="norm" else bayesian_type("2d","norm")
    elif layer in [Conv2dFlipout,Conv2dReparameterization]:
        return layer if layer_type=="bays" else nn.Conv2d
    elif layer == nn.ConvTranspose2d:
        return layer if layer_type=="norm" else bayesian_type("2d","tran")
    elif layer in [ConvTranspose2dFlipout,ConvTranspose2dReparameterization]:
        return layer if layer_type=="bays" else nn.ConvTranspose2d
    elif layer == nn.Conv3d:
        return layer if layer_type=="norm" else bayesian_type("3d","norm")
    elif layer in [Conv3dFlipout,Conv3dReparameterization]:
        return layer if layer_type=="bays" else nn.Conv3d
    elif layer == nn.ConvTranspose3d:
        return layer if layer_type=="norm" else bayesian_type("3d","tran")
    elif layer in [ConvTranspose3dFlipout,ConvTranspose3dReparameterization]:
        return layer if layer_type=="bays" else nn.ConvTranspose3d