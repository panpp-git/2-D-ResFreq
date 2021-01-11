import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2,fftshift,ifft2,fft

def set_pre_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        net = FrequencyRepresentationModule_2D_modified(signal_dim=[args.signal_dim_0, args.signal_dim_1],
                                               n_filters=args.fr_n_filters,
                                               inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                               upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                               kernel_out=args.fr_kernel_out, params=args)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

def set_pre_deepfreq_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        net = FrequencyRepresentationModule_2D_modified_deepfreq(signal_dim=[args.signal_dim_0, args.signal_dim_1],
                                               n_filters=args.fr_n_filters,
                                               inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                               upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                               kernel_out=args.fr_kernel_out, params=args)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

def set_pre_skip_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        net = FrequencyRepresentationModule_2D_modified_skip(signal_dim=[args.signal_dim_0, args.signal_dim_1],
                                               n_filters=args.fr_n_filters,
                                               inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                               upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                               kernel_out=args.fr_kernel_out, params=args)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net



class FrequencyRepresentationModule_2D_skip_biggrid(nn.Module):
    def __init__(self, signal_dim=None, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25, params=None):
        super().__init__()

        self.A_num=8
        n_filters=self.A_num
        self.input_layer = nn.Conv2d(2, 128*self.A_num, kernel_size=(1,64), padding=0, bias=False)
        self.input_layer2=nn.Conv2d(self.A_num, 32*self.A_num, kernel_size=(1,8), padding=0, bias=False)
        kernel_size = 5
        self.mid1 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=(4,2),
                                       padding=(1,2), output_padding=1, bias=False)

        kernel_size = 5
        mod=[]
        for i in range(32):
            tmp = []
            tmp += [
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2,
                                bias=False),

                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2,
                              bias=False),
                nn.BatchNorm2d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(32):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        kernel_size =5
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, kernel_size=kernel_size, stride=(4,2),
                                            padding=(1,2), output_padding=1, bias=False)




    def forward(self, inp):
        bsz = inp.size(0)
        signal_dim_0 = inp.size(2)
        signal_dim_1 = inp.size(3)
        channal=inp.size(1)

        x = self.input_layer(inp)
        x=x.squeeze(-1).view(bsz,self.A_num,128,-1)
        x = self.input_layer2(x)
        x = x.squeeze(-1).view(bsz, self.A_num, 32, -1)
        x=self.mid1(x)

        for i in range(32):
            res_x=self.mod[i](x)
            x = res_x + x
            x=self.activate_layer[i](x)
        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x




class FrequencyRepresentationModule_2D_modified_skip(nn.Module):
    def __init__(self, signal_dim=None, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25, params=None):
        super().__init__()

        self.A_num=8
        n_filters=self.A_num
        self.input_layer = nn.Conv2d(2, 128*self.A_num, kernel_size=(1,64), padding=0, bias=False)
        self.input_layer2=nn.Conv2d(self.A_num, 16*self.A_num, kernel_size=(1,8), padding=0, bias=False)
        # self.input_layer3 = nn.Conv2d(self.A_num, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        kernel_size = 5
        self.mid = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=2,
                                       padding=(kernel_size - 2 + 1) // 2, output_padding=1, bias=False)

        kernel_size = 5

        mod=[]
        for i in range(64):
            tmp = []
            tmp += [
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2,
                                bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2,
                          bias=False),
                nn.BatchNorm2d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(64):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        kernel_size =5
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, kernel_size=kernel_size, stride=2,
                                            padding=(kernel_size - 2 + 1) // 2, output_padding=1, bias=False)


    def forward(self, inp):
        bsz = inp.size(0)
        signal_dim_0 = inp.size(2)
        signal_dim_1 = inp.size(3)
        channal=inp.size(1)

        x = self.input_layer(inp)
        x=x.squeeze(-1).view(bsz,self.A_num,128,-1)
        x = self.input_layer2(x)
        x = x.squeeze(-1).view(bsz, self.A_num, 16, -1)
        x=self.mid(x)

        for i in range(64):
            res_x=self.mod[i](x)
            x = res_x + x
            x=self.activate_layer[i](x)

        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x






class FrequencyRepresentationModule_2D_modified(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_280-snr-15-30-kernel5-batch64-frlog10-1-ampnormal-anum8.pth'
    def __init__(self, signal_dim=None, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25, params=None):
        super().__init__()

        self.A_num=8
        n_filters=self.A_num
        self.input_layer = nn.Conv2d(2, 128*self.A_num, kernel_size=(1,64), padding=0, bias=False)
        self.input_layer2=nn.Conv2d(self.A_num, 16*self.A_num, kernel_size=(1,8), padding=0, bias=False)
        # self.input_layer3 = nn.Conv2d(self.A_num, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.upsampling = upsampling
        self.mid1= nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=2,
                                            padding=(kernel_size - 2 + 1) // 2, output_padding=1, bias=False)
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU()
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, kernel_size=kernel_size, stride=2,
                                            padding=(kernel_size - 2 + 1) // 2, output_padding=1, bias=False)



    def forward(self, inp):
        bsz = inp.size(0)
        signal_dim_0 = inp.size(2)
        signal_dim_1 = inp.size(3)
        channal=inp.size(1)
        x = self.input_layer(inp)
        x=x.squeeze(-1).view(bsz,self.A_num,128,-1)
        x = self.input_layer2(x)
        x = x.squeeze(-1).view(bsz, self.A_num, 16, -1)
        # x= self.input_layer3(x)
        x=self.mid1(x)
        x = self.mod(x)

        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x


class FrequencyRepresentationModule_2D_modified_deepfreq(nn.Module):
    def __init__(self, signal_dim=None, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25, params=None):
        super().__init__()

        self.A_num=8
        n_filters=self.A_num
        self.input_layer = nn.Conv2d(2, 128*self.A_num, kernel_size=(1,64), padding=0, bias=False)
        self.input_layer2=nn.Conv2d(self.A_num, 16*self.A_num, kernel_size=(1,8), padding=0, bias=False)
        self.upsampling = upsampling
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU()
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, kernel_size=kernel_size, stride=4,
                                            padding=(kernel_size - 4 + 1) // 2, output_padding=1, bias=False)



    def forward(self, inp):
        bsz = inp.size(0)
        signal_dim_0 = inp.size(2)
        signal_dim_1 = inp.size(3)
        channal=inp.size(1)
        x = self.input_layer(inp)
        x=x.squeeze(-1).view(bsz,self.A_num,128,-1)
        x = self.input_layer2(x)
        x = x.squeeze(-1).view(bsz, self.A_num, 16, -1)
        x = self.mod(x)
        x = self.out_layer(x)
        x = x.squeeze(-3)
        return x





