import os
import argparse
import numpy as np
from data import noise_2D_3
from data.data_2D_new3 import gen_signal_mainlobe
import json
import h5py
import torch
import util_2D_3
from numpy.fft import fft2,fftshift,ifft2,fft
import h5py
import matlab
import pickle
import matlab.engine
import matplotlib.pyplot as plt
engine = matlab.engine.start_matlab()
import scipy.io  as sio


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
def peak_func(S,thresh):
    Nsig,Nw=S.shape[0],S.shape[1]
    Sp = np.zeros([Nsig, Nw ])
    Sb=np.zeros([Nsig+2,Nw+2])
    Sbp=np.zeros([Nsig+2,Nw+2])
    for m in range(Nsig):
        for k in range(Nw):
            Sb[m+1,k+1]=S[m,k]
    for m in range(1,Nsig+1):
        for k in range(1,Nw+1):
            if Sb[m,k]>=np.max(np.array([Sb[m+1,k-1], Sb[m+1,k], Sb[m+1,k+1],Sb[m,k-1] ,
                                         Sb[m,k+1], Sb[m-1,k-1], Sb[m-1,k], Sb[m-1,k+1]])) and Sb[m,k]>=thresh:
                Sbp[m,k]=Sb[m,k]
    for m in range(Nsig):
        for k in range(Nw):
            Sp[m,k]=Sbp[m+1,k+1]
    return Sp

def CRLB(nb,np,snr):
    k=np.array(list(range(-nb/2,nb/2)))
    nt=np/2
    a0=10**(snr/20)

    J=np.zeros([4,4])
    J[0,0]=2*nb*np
    J[1,1]=2*a0**2*nb*np
    L1=np.zeros([len(np),len(nb)])
    L2 = np.zeros([len(np), len(nb)])
    for m in range(np):
        L1[m,:]=k**2
        L2[m,:]=(m-nt)*(m-nt)*np.ones(1,len(k))
    J[2,2]=8*np.pi*np.pi*a0*a0*np.sum(L1)
    J[3,3]=8*np.pi*np.pi*a0*a0*np.sum(L2)
    I=np.inv(J)
    vc=np.sqrt(I[2,2])
    rc=np.sqrt(I[3,3])
    return vc,rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--n_test", default=1, type=int,
                        help="Number of signals")
    parser.add_argument('--signal_dim_0', type=int, default=8, help='dimensionof the input signal')
    parser.add_argument('--signal_dim_1', type=int, default=64, help='dimensionof the input signal')
    parser.add_argument("--minimum_separation", default=1., type=float,
                        help="Minimum distance between spikes, normalized by 1/signal_dim")
    parser.add_argument("--max_freq", default=2, type=int,
                        help="Maximum number of frequency, the distribution is uniform between 1 and max_freq")
    parser.add_argument("--distance", default="normal", type=str,
                        help="Distribution type of the inter-frequency distance")
    parser.add_argument("--amplitude", default="normal_floor", type=str,
                        help="Distribution type of the spike amplitude")
    parser.add_argument("--floor_amplitude", default=0.1, type=float,
                        help="Minimum spike amplitude (only used for the normal_floor distribution)")
    parser.add_argument('--dB', nargs='+', default=[],
                        help='additional dB levels')

    parser.add_argument("--numpy_seed", default=105, type=int,
                        help="Numpy seed")
    parser.add_argument("--torch_seed", default=94, type=int,
                        help="Numpy seed")
    parser.add_argument("--th", default=0.2, type=float,
                        help="point interval threhold")

    args = parser.parse_args()
    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)


    pre_path = 'trained_models/epoch_240-skip-conn-nlayer64.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_module, _, _, _, _ = util_2D_3.load(pre_path, 'pre-skip', device)
    pre_module.cpu()
    pre_module.eval()
    x_len = 64
    y_len = 512

    xgrid_0 = np.linspace(-0.5, 0.5, x_len, endpoint=False)
    xgrid_1 = np.linspace(-0.5, 0.5, y_len, endpoint=False)


    s, f, nfreq,r = gen_signal_mainlobe(
        num_samples=args.n_test,
        signal_dim_0=args.signal_dim_0,
        signal_dim_1=args.signal_dim_1,
        num_freq=args.max_freq,
        min_sep=args.minimum_separation,
        distance=args.distance,
        amplitude=args.amplitude,
        floor_amplitude=args.floor_amplitude,
        variable_num_freq=False,th=args.th)


    iter_num=1
    for iter in range(iter_num):
        print(iter)
        noisy_signals = noise_2D_3.noise_torch(torch.tensor(s), 100, 'gaussian').cpu().numpy()
        max_v = np.max(np.sqrt(np.power(noisy_signals[0,0,:,:], 2) + np.power(noisy_signals[0,1,:,:], 2)))
        noisy_signals[0][0] = (noisy_signals[0][0]) / (max_v)
        noisy_signals[0][1] = (noisy_signals[0][1]) / (max_v)
        f = h5py.File('signal_acc.h5', 'w')
        f['signal_acc'] = noisy_signals[0]
        f.close()
        with torch.no_grad():
            y = np.abs(
                fftshift(fft2((noisy_signals[0, 0, :, :] + 1j * noisy_signals[0, 1, :, :]), [64, 512])))
            pre_50dB = pre_module(torch.tensor(noisy_signals[0][None]))

        pre_50dB = pre_50dB.numpy()
        pre_50dB = pre_50dB.squeeze(-3)
        pre_50dB = (np.exp((pre_50dB) / 10) - 1) / 10
        pre_50dB=pre_50dB/np.max((pre_50dB))
        pre_50dB[pre_50dB<0]=0
        # pre_50dB=10*np.log10(pre_50dB+1e-6)
        pre_50dB2 = peak_func(pre_50dB, -150)

        y=y/np.max(y)
        # y=10*np.log10(y+1e-6)
        y2=peak_func(y,-150)

        # music and esprit
        f = h5py.File('tgt_num_acc.h5', 'w')  # 创建一个h5文件，文件指针是f
        f['tgt_num_acc'] = args.max_freq # 将数据写入文件的主键data下面
        f.close()
        out = engine.ref_func_mainlobe()
        f = h5py.File('f_est_capon_acc.h5', 'r')
        f_est_capon = f['f_est_capon_acc'][:].T
        f.close()
        f = h5py.File('f_est_music_acc.h5', 'r')
        f_est_music = f['f_est_music_acc'][:].T
        f.close()

        f_est_capon=f_est_capon/np.max(f_est_capon)
        f_est_capon2 = peak_func(f_est_capon, -150)

        f_est_music=f_est_music/np.max(f_est_music)
        f_est_music2 = peak_func(f_est_music, -150)

    sio.savemat('mat-07-100dB.mat', {'deep': pre_50dB, 'fft': y,'capon':f_est_capon,'music':f_est_music})

    x_num = x_len
    y_num = y_len

    fig=plt.figure(figsize=((10,9)))
    plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.view_init(elev=20,azim=-20)
    ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
    ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
    ax3.set_title('Periodogram',fontsize=32)
    zs_1 = np.linspace(-0.5, 0.5, num=x_num)
    zs_2 = np.linspace(-0.5, 0.5, num=y_num) # i# input
    zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
    plt.tick_params(labelsize=30)
    labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax3.plot_surface(zs_1,zs_2,y.T,  cmap = plt.get_cmap('gist_heat'),rstride=1, cstride=1)
    ax3.set_zlim(-50, 0)
    # ax3.set_xlim(-0.1, 0.1)
    # ax3.set_ylim(-0.1, 0.1)
    # ax3.view_init(elev=0, azim=0)

    # #
    fig=plt.figure(figsize=((10,9)))
    plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
    ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
    ax3.set_title('Capon',fontsize=32)
    ax3.view_init(elev=20,azim=-20)
    zs_1 = np.linspace(-0.5, 0.5, num=x_num)
    zs_2 = np.linspace(-0.5, 0.5, num=y_num) # i# input
    zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
    plt.tick_params(labelsize=30)
    labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax3.plot_surface(zs_1,zs_2,f_est_capon, cmap = plt.get_cmap('gist_heat'),rstride=1, cstride=1)
    ax3.set_zlim(-50, 0)
    # ax3.set_xlim(-0.1, 0.1)
    # ax3.view_init(elev=0, azim=0)
    # ax3.set_ylim(-0.1, 0.1)

    fig=plt.figure(figsize=((10,9)))
    plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
    ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
    ax3.set_title('MUSIC',fontsize=32)
    ax3.view_init(elev=20,azim=-20)
    zs_1 = np.linspace(-0.5, 0.5, num=x_num)
    zs_2 = np.linspace(-0.5, 0.5, num=y_num) # i# input
    zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
    plt.tick_params(labelsize=30)
    labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax3.plot_surface(zs_1,zs_2,f_est_music, cmap = plt.get_cmap('gist_heat'),rstride=1, cstride=1)
    ax3.set_zlim(-50, 0)
    # ax3.set_xlim(-0.1, 0.1)
    # ax3.view_init(elev=0, azim=0)
    # ax3.set_ylim(-0.1, 0.1)

    fig=plt.figure(figsize=((10,9)))
    plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.view_init(elev=20,azim=-20)
    ax3.set_title('2D-ResFreq',fontsize=32)
    ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
    ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
    zs_1 = np.linspace(-0.5, 0.5, num=x_num)
    zs_2 = np.linspace(-0.5, 0.5, num=y_num) # i# input
    zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
    plt.tick_params(labelsize=30)
    labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax3.plot_surface(zs_1,zs_2,pre_50dB.T, cmap = plt.get_cmap('gist_heat'),rstride=1, cstride=1)
    ax3.set_zlim(-50, 0)
    # ax3.view_init(elev=0, azim=0)
    # ax3.set_xlim(-0.1, 0.1)
    # ax3.set_ylim(-0.1, 0.1)
    x=1






