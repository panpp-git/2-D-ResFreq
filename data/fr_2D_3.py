import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def freq2pre(f, xgrid, kernel_type='gaussian', param=None,signal_dim_0=50,r=None):
    """
    Convert an array of frequencies to a frequency representation discretized on xgrid.
    """
    if kernel_type == 'gaussian':
        return gaussian_kernel_pre(f, xgrid, param,signal_dim_0,r)
    elif kernel_type == 'triangle':
        return triangle(f, xgrid, param)

def freq2fr(f, xgrid, kernel_type='gaussian', param=None,r=None):
    """
    Convert an array of frequencies to a frequency representation discretized on xgrid.
    """
    if kernel_type == 'gaussian':
        return gaussian_kernel(f, xgrid, param,r)
    elif kernel_type == 'triangle':
        return triangle(f, xgrid, param)
    elif kernel_type == 'test':
        return gaussian_kernel_test(f, xgrid, param,r)

def gaussian_kernel_pre(f, xgrid, sigma,signal_dim_0,r):
    """
    Create a frequency representation with a Gaussian kernel.
    """
    fr = np.zeros((f.shape[0], signal_dim_0, xgrid[1].shape[0]), dtype='float32')
    occ=np.ones([signal_dim_0,1],dtype='float32')
    for i in range(f.shape[2]):
        dist = np.abs(xgrid[1][None, :] - f[:,1, i][:, None])
        rdist = np.abs(xgrid[1][None, :] - (f[:,1, i][:, None] + 1))
        ldist = np.abs(xgrid[1][None, :] - (f[:,1, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)
        for ii in range(fr.shape[0]):
            if f[ii, 0, i] == -10 or f[ii, 1, i] == -10:
                continue
            fr[ii,:,:]+=(occ*np.exp(- dist[ii,:][:,None] ** 2 / sigma[1] ** 2).T)

    return fr

def gaussian_kernel(f, xgrid, sigma,r):
    fr=np.zeros((f.shape[0],xgrid[0].shape[0],xgrid[1].shape[0]),dtype='float32')

    for i in range(f.shape[2]):
        dist=np.abs(xgrid[0][None, :] - (f[:,0, i][:, None]))
        rdist = np.abs(xgrid[0][None, :] - (f[:,0, i][:, None] + 1))
        ldist = np.abs(xgrid[0][None, :] - (f[:,0, i][:, None] - 1))
        dist=np.minimum(dist,rdist,ldist)
        dist1 = np.abs(xgrid[1][None, :] - f[:, 1, i][:, None])
        rdist = np.abs(xgrid[1][None, :] - (f[:, 1, i][:, None] + 1))
        ldist= np.abs(xgrid[1][None, :] - (f[:, 1, i][:, None] - 1))
        dist1 = np.minimum(dist1, rdist, ldist)

        for ii in range(fr.shape[0]):
            if f[ii, 0, i] == -10 or f[ii, 1, i] == -10:
                continue
            fr[ii,:,:]+=((np.exp(- dist[ii,:][:,None] ** 2 / sigma[0] ** 2))*np.exp(- dist1[ii,:][:,None] ** 2 / sigma[1] ** 2).T)*10*np.log(10*r[ii,i]+1)


    fr_ground = np.ones((f.shape[0], xgrid[0].shape[0], xgrid[1].shape[0]), dtype='float32')
    for ii in range(fr.shape[0]):
        mv=-1
        for k in range(f.shape[2]):
            if f[ii,1,k]==-10:
                break
            if 10*np.log(10*r[ii,k]+1)>mv:
                mv=10*np.log(10*r[ii,k]+1)

        for i in range(f.shape[2]):
            cost=(np.power(mv,2)/np.power(10*np.log(10*r[ii,i]+1),2)).astype('float32')
            if f[ii,1,i]==-10:
                continue
            idx0 = (f[ii, 0, i] + 0.5) / (1 / (np.shape(xgrid[0])[0]))
            idx1=(f[ii,1,i]+0.5)/(1/(np.shape(xgrid[1])[0]))
            ctr0=int(np.round(idx0))
            ctr1=int(np.round(idx1))
            if ctr0 == np.shape(xgrid[0])[0]:
                ctr0=0
            if ctr0==0:
                ctr0_up=np.shape(xgrid[0])[0] - 1
            else:
                ctr0_up=ctr0-1
            if ctr0==np.shape(xgrid[0])[0] - 1:
                ctr0_down=0
            else:
                ctr0_down=ctr0+1

            if ctr1 == np.shape(xgrid[1])[0]:
                ctr1=0
            if ctr1==0:
                ctr1_left=np.shape(xgrid[1])[0] - 1
            else:
                ctr1_left=ctr1-1
            if ctr1==np.shape(xgrid[1])[0] - 1:
                ctr1_right=0
            else:
                ctr1_right=ctr1+1
            fr_ground[ii, ctr0, ctr1] = cost
            fr_ground[ii,ctr0_up,ctr1]=cost
            fr_ground[ii, ctr0_down, ctr1] = cost
            fr_ground[ii, ctr0, ctr1_left] = cost
            fr_ground[ii, ctr0, ctr1_right] = cost

    return fr,fr_ground



def gaussian_kernel_test(f, xgrid, sigma,r):
    fr=np.zeros((f.shape[0],xgrid[0].shape[0],xgrid[1].shape[0]),dtype='float32')
    rr=np.zeros((fr.shape[0],f.shape[2]))
    for i in range(f.shape[2]):
        dist=np.abs(xgrid[0][None, :] - (f[:,0, i][:, None]))
        rdist = np.abs(xgrid[0][None, :] - (f[:,0, i][:, None] + 1))
        ldist = np.abs(xgrid[0][None, :] - (f[:,0, i][:, None] - 1))
        dist=np.minimum(dist,rdist,ldist)
        dist1 = np.abs(xgrid[1][None, :] - f[:, 1, i][:, None])
        rdist = np.abs(xgrid[1][None, :] - (f[:, 1, i][:, None] + 1))
        ldist= np.abs(xgrid[1][None, :] - (f[:, 1, i][:, None] - 1))
        dist1 = np.minimum(dist1, rdist, ldist)

        for ii in range(fr.shape[0]):
            if i==2 and ii==10:
                xx=1
            if f[ii, 0, i] == -10 or f[ii, 1, i] == -10:
                continue
            fr[ii,:,:]+=((np.exp(- dist[ii,:][:,None] ** 2 / sigma[0] ** 2))*np.exp(- dist1[ii,:][:,None] ** 2 / sigma[1] ** 2).T)*r[ii,i]
            rr[ii,i]=np.max(((np.exp(- dist[ii,:][:,None] ** 2 / sigma[0] ** 2))*np.exp(- dist1[ii,:][:,None] ** 2 / sigma[1] ** 2).T)*r[ii,i])


    fr_ground = np.zeros((f.shape[0], xgrid[0].shape[0], xgrid[1].shape[0]), dtype='float32')
    for ii in range(fr.shape[0]):
        for i in range(f.shape[2]):
            if f[ii,1,i]==-10:
                continue
            idx=(f[ii,1,i]+0.5)/(1/(np.shape(xgrid[1])[0]-1))
            left=int(np.floor(idx))
            right=int(np.ceil(idx))
            fr_ground[ii, :, left] = 1
            fr_ground[ii, :, right] = 1
            if left==0:
                left=np.shape(xgrid[1])[0]-1
            else:
                left-=1
            if right==np.shape(xgrid[1])[0]-1:
                right=0
            else:
                right+=1
            fr_ground[ii, :, left] = 1
            fr_ground[ii, :, right] = 1
    return fr,fr_ground,rr




def triangle(f, xgrid, slope):
    """
    Create a frequency representation with a triangle kernel.
    """
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    for i in range(f.shape[1]):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)
        fr += np.clip(1 - slope * dist, 0, 1)
    return fr

def find_freq_2D(fr, nfreq, xgrid, max_freq=10):
    """
    Extract frequencies from a frequency representation by locating the highest peaks.
    """
    ff = -np.ones((nfreq.shape[0], 2,max_freq))
    for n in range(len(nfreq)):
        if nfreq[n] < 1:  # at least one frequency
            nf = 1
        else:
            nf = nfreq[n]
        tmp1,tmp2=find_peaks_2D(fr[n])
        num_spikes = min(len(tmp1), int(nf))
        ff[n, 0, :num_spikes],ff[n,1, :num_spikes]=tmp1[:num_spikes],tmp2[:num_spikes]

    return ff

def find_peaks_2D(x,xgrid):
    row=len(x)
    col=len(x[0])
    input_padding=np.zeros([row+2,col+2])*-1
    input_padding[1:row+1,1:col+1]=x
    f_0=[]
    f_1=[]
    v=[]
    for i in range(1,row+1):
        for j in range(1,col+1):
            if input_padding[i,j]>input_padding[i-1,j] and input_padding[i,j]>input_padding[i+1,j] and input_padding[i,j]>input_padding[i,j+1] and input_padding[i,j]>input_padding[i,j-1]:
                v.append(input_padding[i,j])
                f_0.append(i)
                f_1.append(j)
    arg=np.argsort(v)
    tmp0=[]
    tmp=[]
    for c in range(len(arg)):
        tmp0.append(f_0[len(arg)-1-c])
        tmp.append(f_1[len(arg)-1-c])
    return xgrid[0][np.array(tmp0)],xgrid[1][np.array(tmp)]

# x=np.array([[1,1,1,1],[1,5,1,1],[1,1,1,1],[1,1,6,1]])
# find_peaks_2D(x)

def find_freq(fr, nfreq, xgrid, max_freq=10):
    """
    Extract frequencies from a frequency representation by locating the highest peaks.
    """
    ff = -np.ones((nfreq.shape[0], max_freq))
    for n in range(len(nfreq)):

        if nfreq[n] < 1:  # at least one frequency
            nf = 1
        else:
            nf = nfreq[n]

        find_peaks_out = scipy.signal.find_peaks(fr[n], height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), int(nf))
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        ff[n, :num_spikes] = np.sort(xgrid[find_peaks_out[0][idx]])
    return ff


def periodogram(signal, xgrid):
    """
    Compute periodogram.
    """
    js = np.arange(signal.shape[1])
    return (np.abs(np.exp(-2.j * np.pi * xgrid[:, None] * js).dot(signal.T) / signal.shape[1]) ** 2).T


def make_hankel(signal, m):
    """
    Auxiliary function used in MUSIC.
    """
    n = len(signal)
    h = np.zeros((m, n - m + 1), dtype='complex128')
    for r in range(m):
        for c in range(n - m + 1):
            h[r, c] = signal[r + c]
    return h


def music(signal, xgrid, nfreq, m=20):
    """
    Compute frequency representation obtained with MUSIC.
    """
    music_fr = np.zeros((signal.shape[0], len(xgrid)))
    for n in range(signal.shape[0]):
        hankel = make_hankel(signal[n], m)
        _, _, V = np.linalg.svd(hankel)
        v = np.exp(-2.0j * np.pi * np.outer(xgrid[:, None], np.arange(0, signal.shape[1] - m + 1)))
        u = V[nfreq[n]:]
        fr = -np.log(np.linalg.norm(np.tensordot(u, v, axes=(1, 1)), axis=0) ** 2)
        music_fr[n] = fr
    return music_fr
