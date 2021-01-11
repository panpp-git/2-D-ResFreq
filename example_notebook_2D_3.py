#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import util_2D_3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data import fr_2D_3
from numpy.fft import fft2,fftshift

import scipy.io as sio

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
# In[2]:

pre_path1 = 'train_models/epoch_240-skip-conn-nlayer64.pth'
data_dir = 'data_2D_test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
pre_module, _, _, _, _ = util_2D_3.load(pre_path1, 'pre-skip', device)
pre_module.cpu()
pre_module.eval()


#save all weights for Fig3, use filters/main.m to plot figures;
fig=plt.figure(figsize=(11,8))
w2=pre_module.input_layer2.weight[:,:,0,:].detach().cpu().numpy()
w1=pre_module.input_layer.weight[:,:,0,:].detach().cpu().numpy()
w1=w1[:,0,:]+1j*w1[:,1,:]
w2=w2[:,0,:]+1j*w2[:,1,:]
sio.savemat('filters-skip.mat',{'w1':w1,'w2':w2})


x_len=64
y_len=512
xgrid_0 = np.linspace(-0.5, 0.5, x_len, endpoint=False)
xgrid_1 = np.linspace(-0.5, 0.5, y_len, endpoint=False)


#load data
f = np.load(os.path.join(data_dir, 'f.npy'))
r = np.load(os.path.join(data_dir, 'r.npy'))
kernel_param_0 = 0.12/ 8
kernel_param_1 = 0.12 / 64
kernel_param = [kernel_param_0, kernel_param_1]
test,fr_ground,rr=fr_2D_3.freq2fr(f, [xgrid_0,xgrid_1], 'test', param=kernel_param,r=r)

signal_50dB = np.load(os.path.join(data_dir, '20.0dB.npy'))
signal_0dB = np.load(os.path.join(data_dir, '0.0dB.npy'))
nfreq =  np.sum(f[:,0] >= -0.5, axis=1)

idx =2

win1 = np.hamming(8)[:, None]
win2 = np.hamming(64)[None, :]
win = win1 * win2
inp_win = np.zeros((2,signal_50dB.shape[2], signal_50dB.shape[3]))
inp_win[0,:,:] = signal_50dB[idx, 0, :, :] * win
inp_win[1,:,:] = signal_50dB[idx, 1, :, :] * win
# In[7]:
max_v = np.max(np.sqrt(np.power(signal_50dB[idx][0], 2) + np.power(signal_50dB[idx][1], 2)))
signal_50dB[idx][0] = (signal_50dB[idx][0]) / (max_v)
signal_50dB[idx][1] =(signal_50dB[idx][1]) / (max_v)

max_v = np.max(np.sqrt(np.power(signal_0dB[idx][0], 2) + np.power(signal_0dB[idx][1], 2)))
signal_0dB[idx][0] = (signal_0dB[idx][0]) / (max_v)
signal_0dB[idx][1] =(signal_0dB[idx][1]) / (max_v)
with torch.no_grad():
    y=np.abs(fftshift(fft2((signal_50dB[idx,0,:,:]+1j*signal_50dB[idx,1,:,:]),[x_len,y_len])))
    y_win=np.abs(fftshift(fft2((inp_win[0,:,:]+1j*inp_win[1,:,:]),[x_len,y_len])))
    pre_50dB=pre_module(torch.tensor(signal_50dB[idx][None]))
    pre_0dB = pre_module(torch.tensor(signal_0dB[idx][None]))
    y0 = np.abs(fftshift(fft2((signal_0dB[idx, 0, :, :] + 1j * signal_0dB[idx, 1, :, :]), [x_len, y_len])))

pre_50dB = pre_50dB.numpy()
pre_50dB=pre_50dB.squeeze(-3)
pre_0dB = pre_0dB.numpy()
pre_0dB=pre_0dB.squeeze(-3)

fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111)

plt.tick_params(labelsize=32)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax.set_xlabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=32)
ax.set_ylabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=32)
ax.imshow(pre_50dB,extent=[-0.5,0.5,0.5,-0.5],cmap='gray_r')


fig = plt.figure(figsize=(10,9))
plt.subplots_adjust(left=0.12,bottom=0.12,right=0.88,top=0.88,wspace=0.01,hspace=0.3)
ax = fig.add_subplot(111)


plt.tick_params(labelsize=36)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
bbox = patches.Rectangle((-0.31, 0.12), 0.04, 0.24, fill=0,edgecolor = 'r')
ax.add_patch(bbox)
plt.text(f[idx, 1, 7] - 0.12, f[idx, 0, 7] + 0.04, s='%s%d' % ('p', 8), verticalalignment='bottom',
         fontsize=36)

ax.set_xlabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36)
ax.set_ylabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36)
ax.imshow(y_win,extent=[-0.5,0.5,0.5,-0.5],cmap='gray_r')
plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig1_c.eps')


fig = plt.figure(figsize=(10,9))
plt.subplots_adjust(left=0.12,bottom=0.12,right=0.88,top=0.88,wspace=0.01,hspace=0.3)
ax = fig.add_subplot(111)
plt.tick_params(labelsize=36)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36)
ax.set_ylabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36)
bbox = patches.Rectangle((f[idx, 1, 7] - 0.02, f[idx, 0, 7] - 0.025), 0.04, 0.06, fill=0,edgecolor = 'r')
ax.add_patch(bbox)
plt.text(f[idx, 1, 7] - 0.12, f[idx, 0, 7] + 0.04, s='%s%d' % ('p', 8), verticalalignment='bottom',
         fontsize=36)
bbox = patches.Rectangle((f[idx,1,3]-0.02, f[idx,0,3]+0.38),0.04, 0.45,fill=0)
ax.add_patch(bbox)
plt.annotate("recurrent sidelodes", xy=(-0.27,0.33), xytext=(-0.2, 0.4), arrowprops=dict(arrowstyle='->'),fontsize=36)
bbox = patches.Rectangle((-0.2, 0.06),0.38, 0.05,fill=0)
ax.add_patch(bbox)
plt.annotate("recurrent sidelodes", xy=(0, 0.12), xytext=(-0.2, 0.4), arrowprops=dict(arrowstyle='->'),fontsize=36)
plt.imshow(y,extent=[-0.5,0.5,0.5,-0.5],cmap='gray_r')
plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig1_b.eps')

fig = plt.figure(figsize=(10,9))
plt.subplots_adjust(left=0.12,bottom=0.12,right=0.88,top=0.88,wspace=0.01,hspace=0.3)
ax = fig.add_subplot(111)

plt.tick_params(labelsize=36)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36)
ax.set_ylabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36)
for i in range(8):
    ind_list=[1,3,4,2,5,6,7,8]
    bbox = patches.Rectangle((f[idx,1,i]-0.02, f[idx,0,i]-0.025),0.04, 0.06,fill=0)
    ax.add_patch(bbox)
    plt.text(f[idx, 1, i]-0.12, f[idx, 0, i]+0.04, s='%s%d' % ('p',ind_list[i]), verticalalignment='bottom',fontsize=36)


plt.imshow(test[idx],extent=[-0.5,0.5,0.5,-0.5],cmap='gray_r')
plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig1_a.eps')



import h5py
import matlab
import matlab.engine
engine = matlab.engine.start_matlab()
f = h5py.File('m_signal.h5', 'w')
f['m_signal'] = signal_50dB[idx]
f.close()
f = h5py.File('music_num.h5', 'w')
f['music_num'] = nfreq[idx]
f.close()
out=engine.ref_music()
f = h5py.File('music.h5', 'r')
f_music = f['music'][:].T
f.close()
f = h5py.File('capon.h5', 'r')
f_capon = f['capon'][:].T
f.close()

f = h5py.File('m_signal.h5', 'w')
f['m_signal'] = signal_0dB[idx]
f.close()
f = h5py.File('music_num.h5', 'w')
f['music_num'] = nfreq[idx]
f.close()
out=engine.ref_music()
f = h5py.File('music.h5', 'r')
f_music0 = f['music'][:].T
f.close()
f = h5py.File('capon.h5', 'r')
f_capon0 = f['capon'][:].T
f.close()


x_num=x_len
y_num=y_len

test2=test[2]/np.max(test[2])
test2 = peak_func(test2, 0)
est_test_idx = np.ones([2, 8]) * (-10.0)
(est_test_idx[0], est_test_idx[1]) = largest_indices(test2,8)

pre_50dB=(np.exp(pre_50dB/10)-1)/10
pre_50dB=pre_50dB/np.max(pre_50dB)
pre_50dB1 = peak_func(pre_50dB, 0)
est_f_idx = np.ones([2, 8]) * (-10.0)
(est_f_idx[0], est_f_idx[1]) = largest_indices(pre_50dB1,8)

y=y/np.max(y)
y1 = peak_func(y, 0)
est_fft_idx = np.ones([2, 8]) * (-10.0)
(est_fft_idx[0], est_fft_idx[1]) = largest_indices(y1,8)


f_capon=f_capon.T/np.max(f_capon)
f_capon1 = peak_func(f_capon, 0)
est_capon_idx = np.ones([2, 8]) * (-10.0)
(est_capon_idx[0], est_capon_idx[1]) = largest_indices(f_capon1,8)

f_music=f_music.T/np.max(f_music)
f_music1 = peak_func(f_music, 0)
est_music_idx = np.ones([2, 8]) * (-10.0)
(est_music_idx[0], est_music_idx[1]) = largest_indices(f_music1,8)


pre_0dB=(np.exp(pre_0dB/10)-1)/10
pre_0dB=pre_0dB/np.max(pre_0dB)
pre_0dB1 = peak_func(pre_0dB, 0)
est_f_idx0 = np.ones([2, 8]) * (-10.0)
(est_f_idx0[0], est_f_idx0[1]) = largest_indices(pre_0dB1,8)

y0=y0/np.max(y0)
y10 = peak_func(y0, 0)
est_fft_idx0 = np.ones([2, 8]) * (-10.0)
(est_fft_idx0[0], est_fft_idx0[1]) = largest_indices(y10,8)


f_capon0=f_capon0.T/np.max(f_capon0)
f_capon10 = peak_func(f_capon0, 0)
est_capon_idx0 = np.ones([2, 8]) * (-10.0)
(est_capon_idx0[0], est_capon_idx0[1]) = largest_indices(f_capon10,8)

f_music0=f_music0.T/np.max(f_music0)
f_music10 = peak_func(f_music0, 0)
est_music_idx0 = np.ones([2, 8]) * (-10.0)
(est_music_idx0[0], est_music_idx0[1]) = largest_indices(f_music10,8)

fig = plt.figure(figsize=((10,9)))
plt.subplots_adjust(left=0.02,bottom=None,right=0.98,top=None,wspace=0.01,hspace=0.3)

ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(423)
ax3 = fig.add_subplot(425)
ax4 = fig.add_subplot(427)
ax5 = fig.add_subplot(422)
ax6 = fig.add_subplot(424)
ax7 = fig.add_subplot(426)
ax8 = fig.add_subplot(428)
kept_num=6


for i in range(8):
    ax1.axvline(-0.5+1/64*int(est_test_idx[0][i]),color='r', linewidth=1, ymin=0, ymax=1.2)
    ax1.set_title('Periodogram, SNR=20dB', fontsize=18)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.axvline(-0.5+1/64*int(est_test_idx[0][i]), color='r', linewidth=1, ymin=0, ymax=1.2)
    ax2.set_title('Capon, SNR=20dB', fontsize=18)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.axvline(-0.5+1/64*int(est_test_idx[0][i]), color='r', linewidth=1, ymin=0, ymax=1.2)
    ax3.set_title('MUSIC, SNR=20dB', fontsize=18)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.axvline(-0.5+1/64*int(est_test_idx[0][i]), color='r', linewidth=1, ymin=0, ymax=1.2)
    ax4.set_title('2D-ResFreq, SNR=20dB', fontsize=18)
    ax4.set_yticks([])
    labels = ax4.get_xticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax4.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=16)
    ax4.tick_params(labelsize=18)
    ax5.axvline(-0.5+1/64*int(est_test_idx[0][i]),color='r', linewidth=1, ymin=0, ymax=1.2)
    ax5.set_title('Periodogram, SNR=0dB', fontsize=18)
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax6.axvline(-0.5+1/64*int(est_test_idx[0][i]), color='r', linewidth=1, ymin=0, ymax=1.2)
    ax6.set_title('Capon, SNR=0dB', fontsize=18)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax7.axvline(-0.5+1/64*int(est_test_idx[0][i]), color='r', linewidth=1, ymin=0, ymax=1.2)
    ax7.set_title('MUSIC, SNR=0dB', fontsize=18)
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax8.axvline(-0.5+1/64*int(est_test_idx[0][i]), color='r', linewidth=1, ymin=0, ymax=1.2)
    ax8.set_title('2D-ResFreq, SNR=0dB', fontsize=18)
    ax8.set_yticks([])
    ax8.tick_params(labelsize=18)
    labels = ax8.get_xticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax8.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=18)
    xaxistick=np.linspace(-0.5, 0.5, 64, endpoint=False)[0:54]
    for ii in range(8):
        if np.abs(est_test_idx[0,i]-est_f_idx[0][ii])<2.5 and np.abs(est_test_idx[1,i]-est_f_idx[1][ii])<2.5:
            tmp=np.zeros(54)
            tmp[int(est_f_idx[0][ii]-kept_num):int(est_f_idx[0][ii]+kept_num)]=pre_50dB[int(est_f_idx[0][ii]-kept_num):int(est_f_idx[0][ii]+kept_num),int(est_f_idx[1][ii])]
            ax4.plot(xaxistick,tmp,'--',linewidth=2,c='k')
            break
    for ii in range(8):
        if np.abs(est_test_idx[0, i] - est_capon_idx[0][ii]) < 2.5 and np.abs(est_test_idx[1, i] - est_capon_idx[1][ii]) < 2.5:
            tmp = np.zeros(54)
            tmp[int(est_capon_idx[0][ii])-kept_num:int(est_capon_idx[0][ii])+kept_num] = f_capon[int(est_capon_idx[0][ii])-kept_num:int(est_capon_idx[0][ii])+kept_num, int(est_capon_idx[1][ii])]
            ax2.plot(xaxistick,tmp, '--',linewidth=2,c='k')
            break
    for ii in range(8):
        if np.abs(est_test_idx[0, i] - est_music_idx[0][ii]) < 2.5 and np.abs(est_test_idx[1, i] - est_music_idx[1][ii]) < 2.5:
            tmp=np.zeros(54)
            tmp[int(est_music_idx[0][ii])-kept_num:int(est_music_idx[0][ii])+kept_num]=f_music[int(est_music_idx[0][ii])-kept_num:int(est_music_idx[0][ii])+kept_num, int(est_music_idx[1][ii])]
            ax3.plot(xaxistick,tmp, '--',linewidth=2,c='k')
            break
    for ii in range(8):
        if np.abs(est_test_idx[0, i] - est_fft_idx[0][ii]) < 2.5 and np.abs(est_test_idx[1, i] - est_fft_idx[1][ii]) < 2.5:
            tmp=np.zeros(54)
            tmp[int(est_fft_idx[0][ii])-kept_num:int(est_fft_idx[0][ii])+kept_num]=y[int(est_fft_idx[0][ii])-kept_num:int(est_fft_idx[0][ii])+kept_num, int(est_fft_idx[1][ii])]
            ax1.plot(xaxistick,tmp, '--',linewidth=2,c='k')
            break
    for ii in range(8):
        if np.abs(est_test_idx[0,i]-est_f_idx0[0][ii])<2.5 and np.abs(est_test_idx[1,i]-est_f_idx0[1][ii])<2.5:
            tmp=np.zeros(54)
            tmp[int(est_f_idx0[0][ii]-kept_num):int(est_f_idx0[0][ii]+kept_num)]=pre_0dB[int(est_f_idx0[0][ii]-kept_num):int(est_f_idx0[0][ii]+kept_num),int(est_f_idx0[1][ii])]
            ax8.plot(xaxistick,tmp,'--',linewidth=2,c='k')
            break
    for ii in range(8):
        if np.abs(est_test_idx[0, i] - est_capon_idx0[0][ii]) < 2.5 and np.abs(est_test_idx[1, i] - est_capon_idx0[1][ii]) < 2.5:
            tmp = np.zeros(54)
            tmp[int(est_capon_idx0[0][ii])-kept_num:int(est_capon_idx0[0][ii])+kept_num] = f_capon0[int(est_capon_idx0[0][ii])-kept_num:int(est_capon_idx0[0][ii])+kept_num, int(est_capon_idx0[1][ii])]
            ax6.plot(xaxistick,tmp, '--',linewidth=2,c='k')
            break
    for ii in range(8):
        if np.abs(est_test_idx[0, i] - est_music_idx0[0][ii]) < 2.5 and np.abs(est_test_idx[1, i] - est_music_idx0[1][ii]) < 2.5:
            tmp=np.zeros(54)
            tmp[int(est_music_idx0[0][ii])-kept_num:int(est_music_idx0[0][ii])+kept_num]=f_music0[int(est_music_idx0[0][ii])-kept_num:int(est_music_idx0[0][ii])+kept_num, int(est_music_idx0[1][ii])]
            ax7.plot(xaxistick,tmp, '--',linewidth=2,c='k')
            break
    for ii in range(8):
        if np.abs(est_test_idx[0, i] - est_fft_idx0[0][ii]) < 2.5 and np.abs(est_test_idx[1, i] - est_fft_idx0[1][ii]) < 2.5:
            tmp=np.zeros(54)
            tmp[int(est_fft_idx0[0][ii])-kept_num:int(est_fft_idx0[0][ii])+kept_num]=y0[int(est_fft_idx0[0][ii])-kept_num:int(est_fft_idx0[0][ii])+kept_num, int(est_fft_idx0[1][ii])]
            ax5.plot(xaxistick,tmp, '--',linewidth=2,c='k')
            break

plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig4_2.eps')



fig = plt.figure(figsize=((10,9)))
ax3 = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_title(r"Ground Truth",fontsize=32)

ax3.view_init(elev=20,azim=-60)
zs_2 = np.linspace(-0.5, 0.5, num=y_num) #
zs_1 = np.linspace(-0.5, 0.5, num=x_num)
zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
plt.tick_params(labelsize=30)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax3.plot_surface(zs_1,zs_2,test[idx].T/np.max(test[idx]), cmap = plt.get_cmap('gist_heat') ,rstride=1, cstride=1)
plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig4_a.eps')

fig=plt.figure(figsize=((10,9)))
plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
ax3 = fig.add_subplot(111, projection='3d')
ax3.view_init(elev=20,azim=-60)
ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_title('Periodogram, SNR=0dB',fontsize=32)
zs_1 = np.linspace(-0.5, 0.5, num=x_num)
zs_2 = np.linspace(-0.5, 0.5, num=y_num) # i# input
zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
plt.tick_params(labelsize=30)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax3.plot_surface(zs_1,zs_2,y.T/np.max(y),  cmap = plt.get_cmap('gist_heat'),rstride=1, cstride=1)
plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig4_f.eps')

# #
fig=plt.figure(figsize=((10,9)))
plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
ax3 = fig.add_subplot(111, projection='3d')
ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_title('Capon, SNR=0dB',fontsize=32)
ax3.view_init(elev=20,azim=-60)
zs_1 = np.linspace(-0.5, 0.5, num=x_num)
zs_2 = np.linspace(-0.5, 0.5, num=y_num) # i# input
zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
plt.tick_params(labelsize=30)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax3.plot_surface(zs_1,zs_2,f_capon/np.max(f_capon), cmap = plt.get_cmap('gist_heat'),rstride=1, cstride=1)
plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig4_g.eps')
#
#
fig=plt.figure(figsize=((10,9)))
plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
ax3 = fig.add_subplot(111, projection='3d')
ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_title('MUSIC, SNR=0dB',fontsize=32)
ax3.view_init(elev=20,azim=-60)
zs_1 = np.linspace(-0.5, 0.5, num=x_num)
zs_2 = np.linspace(-0.5, 0.5, num=y_num) # i# input
zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
plt.tick_params(labelsize=30)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax3.plot_surface(zs_1,zs_2,f_music/np.max(f_music), cmap = plt.get_cmap('gist_heat'),rstride=1, cstride=1)
plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig4_h.eps')

#
fig=plt.figure(figsize=((10,9)))
plt.subplots_adjust(left=0.01,bottom=0.02,right=0.99,top=0.98,wspace=0.01,hspace=0.3)
ax3 = fig.add_subplot(111, projection='3d')
ax3.view_init(elev=20,azim=-60)
ax3.set_title('2D-ResFreq, SNR=0dB',fontsize=32)
ax3.set_xlabel(r'${f_{\rm{1}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
ax3.set_ylabel(r'${f_{\rm{2}}}{\rm{/Hz}}$',fontproperties='stsong',fontsize=36,labelpad=23)
zs_1 = np.linspace(-0.5, 0.5, num=x_num)
zs_2 = np.linspace(-0.5, 0.5, num=y_num) # i# input
zs_1, zs_2 = np.meshgrid(zs_1, zs_2)
plt.tick_params(labelsize=30)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()+ax3.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
pre_50dB=(np.exp(pre_50dB/10)-1)/10
ax3.plot_surface(zs_1,zs_2,np.abs(pre_50dB.T/np.max(pre_50dB)), cmap = plt.get_cmap('gist_heat'),rstride=1, cstride=1)
plt.savefig('F:\matlab_proj\DeepFreq_review\J_modi\Fig4_i.eps')





