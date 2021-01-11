#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import numpy as np
import torch
import util_2D_3
import pickle
import matlab.engine
engine = matlab.engine.start_matlab()
from data import fr_2D_3
import matplotlib.pyplot as plt

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




pre_path1 = 'trained_models/epoch_270_deepfreq_amp10.pth'
pre_path2 = 'trained_models/epoch_160_amp10.pth'
pre_path3 = 'trained_models/epoch_140_amplog.pth'
pre_path4 = 'trained_models/epoch_190_amplog_cost.pth'
pre_path5 = 'trained_models/epoch_240-skip-conn-nlayer64.pth'
data_dir = 'data_2D_3'

# In[3]:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# In[4]:

#load models
pre_module, _, _, _, _ = util_2D_3.load(pre_path1, 'pre_deepfreq', device)
pre_module.cpu()
pre_module.eval()

pre_module2, _, _, _, _ = util_2D_3.load(pre_path2, 'pre', device)
pre_module2.cpu()
pre_module2.eval()

pre_module3, _, _, _, _ = util_2D_3.load(pre_path3, 'pre', device)
pre_module3.cpu()
pre_module3.eval()

pre_module4, _, _, _, _ = util_2D_3.load(pre_path4, 'pre', device)
pre_module4.cpu()
pre_module4.eval()

pre_module5, _, _, _, _ = util_2D_3.load(pre_path5, 'pre-skip', device)
pre_module5.cpu()
pre_module5.eval()

x_len=64
y_len=512
xgrid_0 = np.linspace(-0.5, 0.5, x_len, endpoint=False)
xgrid_1 = np.linspace(-0.5, 0.5, y_len, endpoint=False)

# In[5]:
#load data
ff = np.load(os.path.join(data_dir, 'f.npy'))
rr = np.load(os.path.join(data_dir, 'r.npy'))
kernel_param_0 = 0.12/ 8
kernel_param_1 = 0.12 / 64
kernel_param = [kernel_param_0, kernel_param_1]
test,fr_ground,rr=fr_2D_3.freq2fr(ff, [xgrid_0,xgrid_1], 'test', param=kernel_param,r=rr)


db=['-10.0dB.npy','-5.0dB.npy','0.0dB.npy','5.0dB.npy','10.0dB.npy','15.0dB.npy','20.0dB.npy','25.0dB.npy','30.0dB.npy']
cnt_deep1=np.zeros(len(db))
cnt_deep2=np.zeros(len(db))
cnt_deep3=np.zeros(len(db))
cnt_deep4=np.zeros(len(db))
cnt_deep5=np.zeros(len(db))

amp_terr_deep1=np.zeros([len(db),1000])
amp_terr_deep2=np.zeros([len(db),1000])
amp_terr_deep3=np.zeros([len(db),1000])
amp_terr_deep4=np.zeros([len(db),1000])
amp_terr_deep5=np.zeros([len(db),1000])
for db_iter in range(0,len(db)):
    signal_50dB = np.load(os.path.join(data_dir, db[db_iter]))
    amp_err_deep1 = np.zeros(ff.shape[0])
    amp_err_deep2 = np.zeros(ff.shape[0])
    amp_err_deep3 = np.zeros(ff.shape[0])
    amp_err_deep4 = np.zeros(ff.shape[0])
    amp_err_deep5 = np.zeros(ff.shape[0])
    nfreq =  np.sum(ff[:,0,:] >= -0.5, axis=1)
    for idx in range(ff.shape[0]):
        print(db_iter,idx)
        max_v = np.max(np.sqrt(np.power(signal_50dB[idx][0], 2) + np.power(signal_50dB[idx][1], 2)))
        signal_50dB[idx][0] = (signal_50dB[idx][0]) / (max_v)
        signal_50dB[idx][1] =(signal_50dB[idx][1]) / (max_v)

        with torch.no_grad():

            pre_50dB1 = pre_module(torch.tensor(signal_50dB[idx][None]))
            pre_50dB2 = pre_module2(torch.tensor(signal_50dB[idx][None]))
            pre_50dB3 = pre_module3(torch.tensor(signal_50dB[idx][None]))
            pre_50dB4 = pre_module4(torch.tensor(signal_50dB[idx][None]))
            pre_50dB5 = pre_module5(torch.tensor(signal_50dB[idx][None]))

        pre_50dB1 = pre_50dB1.numpy()
        pre_50dB1 = pre_50dB1.squeeze(-3)


        pre_50dB2 = pre_50dB2.numpy()
        pre_50dB2 = pre_50dB2.squeeze(-3)


        pre_50dB3 = pre_50dB3.numpy()
        pre_50dB3 = pre_50dB3.squeeze(-3)

        pre_50dB4 = pre_50dB4.numpy()
        pre_50dB4 = pre_50dB4.squeeze(-3)
        #
        pre_50dB5 = pre_50dB5.numpy()
        pre_50dB5 = pre_50dB5.squeeze(-3)



        pre_50dB1 = pre_50dB1 / 10
        pre_50dB1_cp = pre_50dB1
        pre_50dB1 = peak_func(pre_50dB1, 0)
        est_f_idx1 = np.ones([2, nfreq[idx]]) * (-10.0)
        (est_f_idx1[0], est_f_idx1[1]) = largest_indices(pre_50dB1, nfreq[idx])
        est_f1 = np.ones([2, nfreq[idx]]) * (-10.0)
        est_f1[0], est_f1[1] = xgrid_0[est_f_idx1[0].astype(int)], xgrid_1[est_f_idx1[1].astype(int)]

        pre_50dB2 = pre_50dB2 / 10
        pre_50dB2_cp = pre_50dB2
        pre_50dB2 = peak_func(pre_50dB2, 0)
        est_f_idx2 = np.ones([2, nfreq[idx]]) * (-10.0)
        (est_f_idx2[0], est_f_idx2[1]) = largest_indices(pre_50dB2, nfreq[idx])
        est_f2 = np.ones([2, nfreq[idx]]) * (-10.0)
        est_f2[0], est_f2[1] = xgrid_0[est_f_idx2[0].astype(int)], xgrid_1[est_f_idx2[1].astype(int)]

        pre_50dB3 = (np.exp(pre_50dB3 / 10) - 1) / 10
        pre_50dB3_cp = pre_50dB3
        pre_50dB3 = peak_func(pre_50dB3, 0)
        est_f_idx3 = np.ones([2, nfreq[idx]]) * (-10.0)
        (est_f_idx3[0], est_f_idx3[1]) = largest_indices(pre_50dB3, nfreq[idx])
        est_f3 = np.ones([2, nfreq[idx]]) * (-10.0)
        est_f3[0], est_f3[1] = xgrid_0[est_f_idx3[0].astype(int)], xgrid_1[est_f_idx3[1].astype(int)]

        pre_50dB4 = (np.exp(pre_50dB4 / 10) - 1) / 10
        pre_50dB4_cp = pre_50dB4
        pre_50dB4 = peak_func(pre_50dB4, 0)
        est_f_idx4 = np.ones([2, nfreq[idx]]) * (-10.0)
        (est_f_idx4[0], est_f_idx4[1]) = largest_indices(pre_50dB4, nfreq[idx])
        est_f4 = np.ones([2, nfreq[idx]]) * (-10.0)
        est_f4[0], est_f4[1] = xgrid_0[est_f_idx4[0].astype(int)], xgrid_1[est_f_idx4[1].astype(int)]
        #
        pre_50dB5 = (np.exp(pre_50dB5 / 10) - 1) / 10
        pre_50dB5_cp = pre_50dB5
        pre_50dB5 = peak_func(pre_50dB5, 0)
        est_f_idx5 = np.ones([2, nfreq[idx]]) * (-10.0)
        (est_f_idx5[0], est_f_idx5[1]) = largest_indices(pre_50dB5, nfreq[idx])
        est_f5 = np.ones([2, nfreq[idx]]) * (-10.0)
        est_f5[0], est_f5[1] = xgrid_0[est_f_idx5[0].astype(int)], xgrid_1[est_f_idx5[1].astype(int)]


        est_r_deep1 = np.zeros(10)
        est_r_deep2 = np.zeros(10)
        est_r_deep3 = np.zeros(10)
        est_r_deep4 = np.zeros(10)
        est_r_deep5 = np.zeros(10)
        for tgt in range(nfreq[idx]):

            judge = (np.abs(est_f1[0, :] - ff[idx, 0, tgt]) < 1 / 6 / 8) * (
                        np.abs(est_f1[1, :] - ff[idx, 1, tgt]) < 1 / 6 / 64)
            if np.sum(judge) >= 1:
                cnt_deep1[db_iter] += 1
                dis = 9999
                for key, value in enumerate(judge):
                    new_dis = np.sqrt(np.sum(np.power(est_f1[:, key] - ff[idx, :, tgt], 2)))
                    if value == 1 and new_dis < dis:
                        dis = new_dis
                        est_r_deep1[tgt] = pre_50dB1[est_f_idx1[0, key].astype(int), est_f_idx1[1, key].astype(int)]

            judge = (np.abs(est_f2[0, :] - ff[idx, 0, tgt]) < 1 / 6 / 8) * (
                        np.abs(est_f2[1, :] - ff[idx, 1, tgt]) < 1 / 6 / 64)
            if np.sum(judge) >= 1:
                cnt_deep2[db_iter] += 1
                dis = 9999
                for key, value in enumerate(judge):
                    new_dis = np.sqrt(np.sum(np.power(est_f2[:, key] - ff[idx, :, tgt], 2)))
                    if value == 1 and new_dis < dis:
                        dis = new_dis
                        est_r_deep2[tgt] = pre_50dB2[est_f_idx2[0, key].astype(int), est_f_idx2[1, key].astype(int)]

            judge = (np.abs(est_f3[0, :] - ff[idx, 0, tgt]) < 1 / 6 / 8) * (
                        np.abs(est_f3[1, :] - ff[idx, 1, tgt]) < 1 / 6 / 64)
            if np.sum(judge) >= 1:
                cnt_deep3[db_iter] += 1
                dis = 9999
                for key, value in enumerate(judge):
                    new_dis = np.sqrt(np.sum(np.power(est_f3[:, key] - ff[idx, :, tgt], 2)))
                    if value == 1 and new_dis < dis:
                        dis = new_dis
                        est_r_deep3[tgt] = pre_50dB3[est_f_idx3[0, key].astype(int), est_f_idx3[1, key].astype(int)]

            judge = (np.abs(est_f4[0, :] - ff[idx, 0, tgt]) < 1 / 6 / 8) * (
                        np.abs(est_f4[1, :] - ff[idx, 1, tgt]) < 1 / 6 / 64)
            if np.sum(judge) >= 1:
                cnt_deep4[db_iter] += 1
                dis = 9999
                for key, value in enumerate(judge):
                    new_dis = np.sqrt(np.sum(np.power(est_f4[:, key] - ff[idx, :, tgt], 2)))
                    if value == 1 and new_dis < dis:
                        dis = new_dis
                        est_r_deep4[tgt] = pre_50dB4[est_f_idx4[0, key].astype(int), est_f_idx4[1, key].astype(int)]


            judge = (np.abs(est_f5[0, :] - ff[idx, 0, tgt]) < 1 / 6 / 8) * (
                        np.abs(est_f5[1, :] - ff[idx, 1, tgt]) < 1 / 6 / 64)
            if np.sum(judge) >= 1:
                cnt_deep5[db_iter] += 1
                dis = 9999
                for key, value in enumerate(judge):
                    new_dis = np.sqrt(np.sum(np.power(est_f5[:, key] - ff[idx, :, tgt], 2)))
                    if value == 1 and new_dis < dis:
                        dis = new_dis
                        est_r_deep5[tgt] = pre_50dB5[est_f_idx5[0, key].astype(int), est_f_idx5[1, key].astype(int)]

        amp_err_deep1[idx] = np.sum(np.power((rr[idx, 0:nfreq[idx]] - est_r_deep1[0:nfreq[idx]]), 2))
        amp_err_deep2[idx] = np.sum(np.power((rr[idx, 0:nfreq[idx]] - est_r_deep2[0:nfreq[idx]]), 2))
        amp_err_deep3[idx] = np.sum(np.power((rr[idx, 0:nfreq[idx]] - est_r_deep3[0:nfreq[idx]]), 2))
        amp_err_deep4[idx] = np.sum(np.power((rr[idx, 0:nfreq[idx]] - est_r_deep4[0:nfreq[idx]]), 2))
        amp_err_deep5[idx] = np.sum(np.power((rr[idx, 0:nfreq[idx]] - est_r_deep5[0:nfreq[idx]]), 2))

    amp_terr_deep1[db_iter, :] = amp_err_deep1
    amp_terr_deep2[db_iter, :] = amp_err_deep2
    amp_terr_deep3[db_iter, :] = amp_err_deep3
    amp_terr_deep4[db_iter, :] = amp_err_deep4
    amp_terr_deep5[db_iter, :] = amp_err_deep5


pickle.dump(amp_terr_deep1, open('2d-deepfreq-amp.txt', 'wb'))
pickle.dump(cnt_deep1, open('2d-deepfreq-cnt.txt', 'wb'))
pickle.dump(amp_terr_deep2, open('preupsmp-amp.txt', 'wb'))
pickle.dump(cnt_deep2, open('preupsmp-cnt.txt', 'wb'))
pickle.dump(amp_terr_deep3, open('preupsmp-amplog-amp.txt', 'wb'))
pickle.dump(cnt_deep3, open('preupsmp-amplog-cnt.txt', 'wb'))
pickle.dump(amp_terr_deep4, open('preupsmp-wgtamp-amp.txt', 'wb'))
pickle.dump(cnt_deep4, open('preupsmp-wgtamp-cnt.txt', 'wb'))
pickle.dump(amp_terr_deep5, open('2d-resfreq-amp.txt', 'wb'))
pickle.dump(cnt_deep5, open('2d-resfreq-cnt.txt', 'wb'))


target_num=5380
db=[-10,-5,0,5,10,15,20,25,30]-10*np.log(np.sqrt(2))  # because the complex noise power need to divide sqrt(2)
fig = plt.figure(figsize=(11,7))

ax = fig.add_subplot(111)
plt.subplots_adjust(left=None,bottom=0.1,right=None,top=0.98,wspace=None,hspace=None)
plt.tick_params(labelsize=16)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax = fig.add_subplot(111)
ax.set_xlabel('SNR / dB',size=20)
ax.set_ylabel('MSE',size=20)

cnt_deep=pickle.load(open('preupsmp-amp.txt', 'rb'))
cnt_deep1=pickle.load(open('preupsmp-amplog-amp.txt', 'rb'))
cnt_deep2=pickle.load(open('preupsmp-wgtamp-amp.txt', 'rb'))
cnt_deep3=pickle.load(open('2d-deepfreq-amp.txt', 'rb'))
cnt_deep9=pickle.load(open('2d-resfreq-amp.txt', 'rb'))


ax.plot(db,np.sum(cnt_deep3,1)/target_num,'--',c='m',marker='o',label='2D-DeepFreq',linewidth=4,markersize=10)
ax.plot(db,np.sum(cnt_deep,1)/target_num,'--',c='g',marker='o',label='PreUpSmp',linewidth=4,markersize=10)
ax.plot(db,np.sum(cnt_deep1,1)/target_num,'--',c='b',marker='o',label='PreUpSmp-Amp',linewidth=4,markersize=10)
ax.plot(db,np.sum(cnt_deep2,1)/target_num,'--',c='orange',marker='o',label='PreUpSmp-WgtAmp',linewidth=4,markersize=10)
ax.plot(db,np.sum(cnt_deep9,1)/target_num,'--',c='k',marker='o',label='2D-ResFreq',linewidth=4,markersize=10)

plt.grid(linestyle='-.')
plt.legend(frameon=True,prop={'size':16})

##############################################################MODEL COM##################################################
target_num=5380
db=[-10,-5,0,5,10,15,20,25,30]-10*np.log(np.sqrt(2))

fig = plt.figure(figsize=(11,7))

plt.subplots_adjust(left=None,bottom=0.1,right=None,top=0.98,wspace=None,hspace=None)
plt.tick_params(labelsize=16)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax = fig.add_subplot(111)
ax.set_xlabel('SNR / dB',size=20)
ax.set_ylabel('FNR / %',size=20)
cnt_deep=pickle.load(open('preupsmp-cnt.txt', 'rb'))
cnt_deep1=pickle.load(open('preupsmp-amplog-cnt.txt', 'rb'))
cnt_deep3=pickle.load(open('2d-deepfreq-cnt.txt', 'rb'))
cnt_deep6=pickle.load(open('preupsmp-wgtamp-cnt.txt', 'rb'))
cnt_deep9=pickle.load(open('2d-resfreq-cnt.txt', 'rb'))


ax.plot(db,(target_num-cnt_deep3)/target_num*100,'--',c='m',marker='o',label='2D-DeepFreq',linewidth=4,markersize=10)
ax.plot(db,(target_num-cnt_deep)/target_num*100,'--',c='g',marker='o',label='PreUpSmp',linewidth=4,markersize=10)
ax.plot(db,(target_num-cnt_deep1)/target_num*100,'--',c='b',marker='o',label='PreUpSmp-Amp',linewidth=4,markersize=10)
ax.plot(db,(target_num-cnt_deep6)/target_num*100,'--',c='orange',marker='o',label='PreUpSmp-WgtAmp',linewidth=4,markersize=10)
ax.plot(db,(target_num-cnt_deep9)/target_num*100,'--',c='k',marker='o',label='2D-ResFreq',linewidth=4,markersize=10)


plt.grid(linestyle='-.')
plt.legend(frameon=True,prop={'size':16})



