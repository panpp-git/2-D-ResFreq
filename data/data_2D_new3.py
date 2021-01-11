import numpy as np
import torch


def frequency_generator(f, nf, min_sep_0,min_sep_1, dist_distribution,if_near,near_num):
    min_sep=[min_sep_0,min_sep_1]
    if dist_distribution == 'random':
        random_freq(f, nf, min_sep)
    elif dist_distribution == 'jittered':
        jittered_freq(f, nf, min_sep)
    elif dist_distribution == 'normal':
        normal_freq(f, nf, min_sep,if_near,near_num)


def random_freq(f, nf, min_sep):
    """
    Generate frequencies uniformly.
    """
    for i in range(nf):
        f_new = np.random.rand() - 1 / 2
        condition = True
        while condition:
            f_new = np.random.rand() - 1 / 2
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def jittered_freq(f, nf, min_sep, jit=1):
    """
    Generate jittered frequencies.
    """
    l, r = -0.5, 0.5 - nf * min_sep * (1 + jit)
    s = l + np.random.rand() * (r - l)
    c = np.cumsum(min_sep * (np.ones(nf) + np.random.rand(nf) * jit))
    f[:nf] = (s + c - min_sep + 0.5) % 1 - 0.5


def normal_freq(f, nf, min_sep,if_near=0,near_num=0):
    """
    Distance between two frequencies follows a normal distribution
    """
    n1=1/min_sep[0]
    n2=1/min_sep[1]
    min_sep[0]=min_sep[0]
    min_sep[1] = min_sep[1]
    f[0,0]= np.random.uniform() - 0.5
    f[1,0]= np.random.uniform() - 0.5
    if if_near==0:
        for i in range(1,nf):
            condition=True
            while condition:
                d_0 = np.random.normal(0.05/(n1))
                f_new_0 = (d_0 + np.sign(d_0) * min_sep[0]/2 + f[0][i - 1] + 0.5) % 1 - 0.5

                d_1 = np.random.normal(0.05/(n2))
                f_new_1 = (d_1 + np.sign(d_1) * min_sep[1]/2 + f[1][i - 1] + 0.5) % 1 - 0.5

                condition1=(np.min(np.abs(f[0,:] - f_new_0)) < min_sep[0] and np.min(np.abs(f[1,:] - f_new_1)) < min_sep[1])
                condition2=(np.min(np.abs(f[0,:] - f_new_0)) < min_sep[0] and np.min(np.abs((f[1,:]-1) - f_new_1)) < min_sep[1])
                condition3=(np.min(np.abs(f[0,:] - f_new_0)) < min_sep[0] and np.min(np.abs((f[1,:]+1) - f_new_1)) < min_sep[1])
                condition4=(np.min(np.abs((f[0,:] - 1) - f_new_0)) < min_sep[0] and np.min(np.abs((f[1,:] - 1) - f_new_1)) < min_sep[1])
                condition5=(np.min(np.abs((f[0,:] - 1) - f_new_0)) < min_sep[0] and np.min(np.abs((f[1,:] + 1) - f_new_1)) < min_sep[1])
                condition6 = (np.min(np.abs((f[0, :] - 1) - f_new_0)) < min_sep[0] and np.min(np.abs(f[1, :] - f_new_1)) < min_sep[1])
                condition7 = (np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(np.abs((f[1, :] - 1) - f_new_1)) < min_sep[1])
                condition8 = (np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(np.abs((f[1, :] + 1) - f_new_1)) < min_sep[1])
                condition9 = (np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(np.abs(f[1, :] - f_new_1)) <min_sep[1])
                condition = condition1 or condition2 or condition3 or condition4 or condition5 or condition6 or condition7 or condition8 or condition9
            f[0][i] =f_new_0
            f[1][i]=f_new_1
    else:
        cnt=0
        for i in range(1, nf):
            if cnt<near_num:
                condition = False
                while not condition:
                    d_0 = np.random.normal(0.05 / n1)
                    f_new_0 = (d_0 + np.sign(d_0) * min_sep[0]/2 + f[0][i - 1] + 0.5) % 1 - 0.5

                    d_1 = np.random.normal(0.05 / n2)
                    f_new_1 = (d_1 + np.sign(d_1) * min_sep[1]/2 + f[1][i - 1] + 0.5) % 1 - 0.5

                    condition1 = (np.min(np.abs(f[0, :] - f_new_0)) < min_sep[0] and np.min(np.abs(f[1, :] - f_new_1)) <
                                  min_sep[1]) and (np.min(np.abs(f[0, :] - f_new_0)) > min_sep[0]/3 and np.min(np.abs(f[1, :] - f_new_1)) >
                                  min_sep[1]/3)
                    condition2 = (np.min(np.abs(f[0, :] - f_new_0)) < min_sep[0] and np.min(np.abs((f[1, :] - 1) - f_new_1)) <
                                min_sep[1]) and (np.min(np.abs(f[0, :] - f_new_0)) > min_sep[0]/3 and np.min(np.abs((f[1, :] - 1) - f_new_1)) >
                                min_sep[1]/3)
                    condition3 = (np.min(np.abs(f[0, :] - f_new_0)) < min_sep[0] and np.min(np.abs((f[1, :] + 1) - f_new_1)) <
                                min_sep[1]) and (np.min(np.abs(f[0, :] - f_new_0)) > min_sep[0]/3 and np.min(np.abs((f[1, :] + 1) - f_new_1)) >
                                min_sep[1]/3)
                    condition4 = (np.min(np.abs((f[0, :] - 1) - f_new_0)) < min_sep[0] and np.min(
                        np.abs((f[1, :] - 1) - f_new_1)) < min_sep[1]) and (np.min(np.abs((f[0, :] - 1) - f_new_0)) > min_sep[0]/3 and np.min(
                        np.abs((f[1, :] - 1) - f_new_1)) > min_sep[1]/3)
                    condition5 = (np.min(np.abs((f[0, :] - 1) - f_new_0)) < min_sep[0] and np.min(
                        np.abs((f[1, :] + 1) - f_new_1)) < min_sep[1]) and (np.min(np.abs((f[0, :] - 1) - f_new_0)) > min_sep[0]/3 and np.min(
                        np.abs((f[1, :] + 1) - f_new_1)) > min_sep[1]/3)
                    condition6 = (np.min(np.abs((f[0, :] - 1) - f_new_0)) < min_sep[0] and np.min(np.abs(f[1, :] - f_new_1)) <
                                min_sep[1]) and (np.min(np.abs((f[0, :] - 1) - f_new_0)) > min_sep[0]/3 and np.min(np.abs(f[1, :] - f_new_1)) >
                                min_sep[1]/3)
                    condition7 = (np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(
                        np.abs((f[1, :] - 1) - f_new_1)) < min_sep[1]) and (np.min(np.abs((f[0, :] + 1) - f_new_0)) > min_sep[0]/3 and np.min(
                        np.abs((f[1, :] - 1) - f_new_1)) > min_sep[1]/3)
                    condition8 = (np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(
                        np.abs((f[1, :] + 1) - f_new_1)) < min_sep[1]) and (np.min(np.abs((f[0, :] + 1) - f_new_0)) > min_sep[0]/3 and np.min(
                        np.abs((f[1, :] + 1) - f_new_1)) > min_sep[1]/3)
                    condition9 = (np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(np.abs(f[1, :] - f_new_1)) <
                                min_sep[1]) and (np.min(np.abs((f[0, :] + 1) - f_new_0)) > min_sep[0]/3 and np.min(np.abs(f[1, :] - f_new_1)) >
                                min_sep[1]/3)
                    condition = condition1 or condition2 or condition3 or condition4 or condition5 or condition6 or condition7 or condition8 or condition9
            else:
                condition = True
                while condition:
                    d_0 = np.random.normal(0.05/ (1 / min_sep[0]))
                    f_new_0 = (d_0 + np.sign(d_0) * min_sep[0]/2 + f[0][i - 1] + 0.5) % 1 - 0.5

                    d_1 = np.random.normal(0.05/ (1 / min_sep[1]))
                    f_new_1 = (d_1 + np.sign(d_1) * min_sep[1]/2 + f[1][i - 1] + 0.5) % 1 - 0.5

                    condition1 = (np.min(np.abs(f[0, :] - f_new_0)) < min_sep[0] and np.min(np.abs(f[1, :] - f_new_1)) <
                                  min_sep[1])
                    condition2 = (
                            np.min(np.abs(f[0, :] - f_new_0)) < min_sep[0] and np.min(np.abs((f[1, :] - 1) - f_new_1)) <
                            min_sep[1])
                    condition3 = (
                            np.min(np.abs(f[0, :] - f_new_0)) < min_sep[0] and np.min(np.abs((f[1, :] + 1) - f_new_1)) <
                            min_sep[1])
                    condition4 = (np.min(np.abs((f[0, :] - 1) - f_new_0)) < min_sep[0] and np.min(
                        np.abs((f[1, :] - 1) - f_new_1)) < min_sep[1])
                    condition5 = (np.min(np.abs((f[0, :] - 1) - f_new_0)) < min_sep[0] and np.min(
                        np.abs((f[1, :] + 1) - f_new_1)) < min_sep[1])
                    condition6 = (
                            np.min(np.abs((f[0, :] - 1) - f_new_0)) < min_sep[0] and np.min(np.abs(f[1, :] - f_new_1)) <
                            min_sep[1])
                    condition7 = (np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(
                        np.abs((f[1, :] - 1) - f_new_1)) < min_sep[1])
                    condition8 = (np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(
                        np.abs((f[1, :] + 1) - f_new_1)) < min_sep[1])
                    condition9 = (
                            np.min(np.abs((f[0, :] + 1) - f_new_0)) < min_sep[0] and np.min(np.abs(f[1, :] - f_new_1)) <
                            min_sep[1])
                    condition = condition1 or condition2 or condition3 or condition4 or condition5 or condition6 or condition7 or condition8 or condition9
            cnt+=1
            f[0][i] = f_new_0
            f[1][i] = f_new_1


def amplitude_generation(dim, amplitude, floor_amplitude=0.1):
    """
    Generate the amplitude associated with each frequency.
    """
    if amplitude == 'uniform':
        return np.random.rand(*dim) * (1.1 - floor_amplitude) + floor_amplitude
    elif amplitude == 'normal':
        return np.abs(np.random.randn(*dim))
    elif amplitude == 'normal_floor':
        return np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == 'alternating':
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(*dim) * np.random.randint(0, 2, size=dim)

def gen_signal_test(num_samples, signal_dim_0,signal_dim_1, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False,th=0.5):
    s = np.zeros((num_samples,2,signal_dim_0, signal_dim_1))
    xgrid_0 = np.arange(signal_dim_0)[:, None]
    xgrid_1=np.arange(signal_dim_1)[:,None]
    f = np.ones((num_samples, 2,num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    d_sep_0 = min_sep / signal_dim_0
    d_sep_1=min_sep/signal_dim_1
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq


    for n in range(num_samples):
        if n%100==0:
            print(n)
        if_near = np.random.rand()
        if if_near > th and nfreq[n] not in [0,1]:
            if_near = 1
            near_num = np.random.randint(1, nfreq[n])
        else:
            if_near = 0
            near_num = 0
        frequency_generator(f[n], nfreq[n], d_sep_0,d_sep_1, distance,if_near,near_num)

        for i in range(nfreq[n]):
            sin = r[n, i] * (np.exp(2j * np.pi * f[n, 0,i] * xgrid_0.T)).T*np.exp( 2j * np.pi * f[n, 1,i] * xgrid_1.T)
            s[n, 0, :, :] = s[n, 0, :, :] + sin.real
            s[n, 1, :, :] = s[n, 1, :, :] + sin.imag
        for x_idx in range(signal_dim_0):
            amp_mean = np.sqrt(np.mean(np.power(s[n, 0, x_idx, :], 2) + np.power(s[n, 1, x_idx, :], 2)))
            s[n, 0, x_idx, :] = s[n, 0, x_idx, :] / amp_mean
            s[n, 1, x_idx, :] = s[n, 1, x_idx, :] / amp_mean

    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r.astype('float32')

def gen_signal(num_samples, signal_dim_0,signal_dim_1, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False,th=0.5):
    s = np.zeros((num_samples,2,signal_dim_0, signal_dim_1))
    xgrid_0 = np.arange(signal_dim_0)[:, None]
    xgrid_1=np.arange(signal_dim_1)[:,None]
    f = np.ones((num_samples, 2,num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    d_sep_0 = min_sep / signal_dim_0
    d_sep_1=min_sep/signal_dim_1
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    for n in range(num_samples):
        if n%100==0:
            print(n)
        if_near = np.random.rand()
        if if_near > th and nfreq[n] not in [0,1]:
            if_near = 1
            near_num = np.random.randint(1, nfreq[n])
        else:
            if_near = 0
            near_num = 0
        frequency_generator(f[n], nfreq[n], d_sep_0,d_sep_1, distance,if_near,near_num)
        for i in range(nfreq[n]):
            sin = r[n, i] * (np.exp(2j * np.pi * f[n, 0,i] * xgrid_0.T)).T*np.exp( 2j * np.pi * f[n, 1,i] * xgrid_1.T)
            s[n, 0, :, :] = s[n, 0, :, :] + sin.real
            s[n, 1, :, :] = s[n, 1, :, :] + sin.imag
        for x_idx in range(signal_dim_0):
            amp_mean = np.sqrt(np.mean(np.power(s[n, 0, x_idx, :], 2) + np.power(s[n, 1, x_idx, :], 2)))
            s[n, 0, x_idx, :] = s[n, 0, x_idx, :] / amp_mean
            s[n, 1, x_idx, :] = s[n, 1, x_idx, :] / amp_mean

    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r.astype('float32')



def gen_signal_resolution(num_samples, signal_dim_0,signal_dim_1, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False,th=0.5,dw=0.1):
    s = np.zeros((num_samples,2,signal_dim_0, signal_dim_1))
    xgrid_0 = np.arange(signal_dim_0)[:, None]
    xgrid_1=np.arange(signal_dim_1)[:,None]
    f = np.ones((num_samples, 2,num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    r[0,0:2]=1
    d_sep_0 = min_sep / signal_dim_0
    d_sep_1=min_sep/signal_dim_1
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq


    for n in range(num_samples):
        if n%100==0:
            print(n)
        if_near = np.random.rand()
        if if_near > th and nfreq[n] not in [0,1]:
            if_near = 1
            near_num = np.random.randint(1, nfreq[n])
        else:
            if_near = 0
            near_num = 0
        frequency_generator(f[n], nfreq[n], d_sep_0,d_sep_1, distance,if_near,near_num)

        f[0, :, 0:2] = np.array([[0,0], [0,dw/64]])
        for i in range(nfreq[n]):
            sin = r[n, i] * (np.exp(2j * np.pi * f[n, 0,i] * xgrid_0.T)).T*np.exp( 2j * np.pi * f[n, 1,i] * xgrid_1.T)
            s[n, 0, :, :] = s[n, 0, :, :] + sin.real
            s[n, 1, :, :] = s[n, 1, :, :] + sin.imag
        for x_idx in range(signal_dim_0):
            amp_mean = np.sqrt(np.mean(np.power(s[n, 0, x_idx, :], 2) + np.power(s[n, 1, x_idx, :], 2)))
            s[n, 0, x_idx, :] = s[n, 0, x_idx, :] / amp_mean
            s[n, 1, x_idx, :] = s[n, 1, x_idx, :] / amp_mean

    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r.astype('float32')



def gen_signal_mainlobe(num_samples, signal_dim_0,signal_dim_1, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False,th=0.5):
    s = np.zeros((num_samples,2,signal_dim_0, signal_dim_1))
    xgrid_0 = np.arange(signal_dim_0)[:, None]
    xgrid_1=np.arange(signal_dim_1)[:,None]
    f = np.ones((num_samples, 2,num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    r[0,0:3]=1
    d_sep_0 = min_sep / signal_dim_0
    d_sep_1=min_sep/signal_dim_1
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq


    for n in range(num_samples):
        if n%100==0:
            print(n)
        if_near = np.random.rand()
        if if_near > th and nfreq[n] not in [0,1]:
            if_near = 1
            near_num = np.random.randint(1, nfreq[n])
        else:
            if_near = 0
            near_num = 0
        frequency_generator(f[n], nfreq[n], d_sep_0,d_sep_1, distance,if_near,near_num)
        # f[0, :, 0:1] = np.array([[0], [0]])
        f[0, :, 0:2] = np.array([[0, 0], [0,  0 + 0.7 / 64]])
        # f[0, :, 0:3] = np.array([[0,0+3/8,0], [0,0,0+3/64]])
        for i in range(nfreq[n]):
            sin = r[n, i] * (np.exp(2j * np.pi * f[n, 0,i] * xgrid_0.T)).T*np.exp( 2j * np.pi * f[n, 1,i] * xgrid_1.T)
            s[n, 0, :, :] = s[n, 0, :, :] + sin.real
            s[n, 1, :, :] = s[n, 1, :, :] + sin.imag
        for x_idx in range(signal_dim_0):
            amp_mean = np.sqrt(np.mean(np.power(s[n, 0, x_idx, :], 2) + np.power(s[n, 1, x_idx, :], 2)))
            s[n, 0, x_idx, :] = s[n, 0, x_idx, :] / amp_mean
            s[n, 1, x_idx, :] = s[n, 1, x_idx, :] / amp_mean

    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r.astype('float32')


def gen_signal_accuracy(num_samples, signal_dim_0,signal_dim_1, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False,th=0.5):
    s = np.zeros((num_samples,2,signal_dim_0, signal_dim_1))
    xgrid_0 = np.arange(signal_dim_0)[:, None]
    xgrid_1=np.arange(signal_dim_1)[:,None]
    f = np.ones((num_samples, 2,num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    r[0,0:1]=1
    d_sep_0 = min_sep / signal_dim_0
    d_sep_1=min_sep/signal_dim_1
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq


    for n in range(num_samples):
        if n%100==0:
            print(n)
        if_near = np.random.rand()
        if if_near > th and nfreq[n] not in [0,1]:
            if_near = 1
            near_num = np.random.randint(1, nfreq[n])
        else:
            if_near = 0
            near_num = 0
        frequency_generator(f[n], nfreq[n], d_sep_0,d_sep_1, distance,if_near,near_num)

        f[0, :, 0:1] = np.array([[0], [0]])
        for i in range(nfreq[n]):
            sin = r[n, i] * (np.exp(2j * np.pi * f[n, 0,i] * xgrid_0.T)).T*np.exp( 2j * np.pi * f[n, 1,i] * xgrid_1.T)
            s[n, 0, :, :] = s[n, 0, :, :] + sin.real
            s[n, 1, :, :] = s[n, 1, :, :] + sin.imag
        for x_idx in range(signal_dim_0):
            amp_mean = np.sqrt(np.mean(np.power(s[n, 0, x_idx, :], 2) + np.power(s[n, 1, x_idx, :], 2)))
            s[n, 0, x_idx, :] = s[n, 0, x_idx, :] / amp_mean
            s[n, 1, x_idx, :] = s[n, 1, x_idx, :] / amp_mean

    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r.astype('float32')
