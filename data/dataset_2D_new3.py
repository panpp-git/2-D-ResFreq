import torch
import numpy as np
import torch.utils.data as data_utils
from data import fr_2D_3, data_2D_new3
from .noise_2D_3 import noise_torch


def load_dataloader(num_samples, signal_dim_0,signal_dim_1, max_n_freq, min_sep, distance, amplitude, floor_amplitude,
                    kernel_type, kernel_param, batch_size, xgrid_0,xgrid_1,th=0.5):
    clean_signals, f, nfreq,r = data_2D_new3.gen_signal(num_samples, signal_dim_0,signal_dim_1, max_n_freq, min_sep, distance=distance,
                                              amplitude=amplitude, floor_amplitude=floor_amplitude,
                                              variable_num_freq=True,th=th)
    fre_representation = fr_2D_3.freq2pre(f, [xgrid_0, xgrid_1], kernel_type, param=kernel_param,signal_dim_0=signal_dim_0,r=r)
    frequency_representation,pre_ground = fr_2D_3.freq2fr(f, [xgrid_0,xgrid_1], kernel_type, param=kernel_param,r=r)

    clean_signals = torch.from_numpy(clean_signals).float()
    f = torch.from_numpy(f).float()
    fre_representation = torch.from_numpy(fre_representation).float()
    frequency_representation = torch.from_numpy(frequency_representation).float()
    pre_ground = torch.from_numpy(pre_ground).float()

    dataset = data_utils.TensorDataset(clean_signals, frequency_representation, f,fre_representation,pre_ground)
    return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_dataloader_fixed_noise(num_samples, signal_dim_0,signal_dim_1, max_n_freq, min_sep, distance, amplitude, floor_amplitude,
                                kernel_type, kernel_param, batch_size,  xgrid_0,xgrid_1, snr, noise,th=0.5):
    clean_signals, f, nfreq,r = data_2D_new3.gen_signal(num_samples,  signal_dim_0,signal_dim_1, max_n_freq, min_sep, distance=distance,
                                              amplitude=amplitude, floor_amplitude=floor_amplitude,
                                              variable_num_freq=True,th=th)
    frequency_representation,pre_ground = fr_2D_3.freq2fr(f, [xgrid_0,xgrid_1], kernel_type, param=kernel_param,r=r)
    fre_representation = fr_2D_3.freq2pre(f, [xgrid_0, xgrid_1], kernel_type, param=kernel_param,signal_dim_0=signal_dim_0,r=r)
    clean_signals = torch.from_numpy(clean_signals).float()
    f = torch.from_numpy(f).float()
    frequency_representation = torch.from_numpy(frequency_representation).float()
    fre_representation = torch.from_numpy(fre_representation).float()
    pre_ground = torch.from_numpy(pre_ground).float()
    noisy_signals = noise_torch(clean_signals, snr, noise)
    dataset = data_utils.TensorDataset(noisy_signals, clean_signals, frequency_representation, f,fre_representation,pre_ground)
    return data_utils.DataLoader(dataset, batch_size=batch_size)


def make_train_data(args):
    xgrid_0 = np.linspace(-0.5, 0.5, args.fr_size_0, endpoint=False)
    xgrid_1=np.linspace(-0.5, 0.5, args.fr_size_1, endpoint=False)
    if args.kernel_type == 'triangle':
        kernel_param = args.triangle_slope / args.signal_dim
    else:
        kernel_param_0 = args.gaussian_std / args.signal_dim_0
        kernel_param_1=args.gaussian_std / args.signal_dim_1
        kernel_param=[kernel_param_0,kernel_param_1]
    return load_dataloader(args.n_training, signal_dim_0=args.signal_dim_0, signal_dim_1=args.signal_dim_1,max_n_freq=args.max_n_freq,
                           min_sep=args.min_sep, distance=args.distance, amplitude=args.amplitude,
                           floor_amplitude=args.floor_amplitude, kernel_type=args.kernel_type,
                           kernel_param=kernel_param, batch_size=args.batch_size, xgrid_0=xgrid_0,xgrid_1=xgrid_1,th=args.th)



def make_eval_data(args):
    xgrid_0 = np.linspace(-0.5, 0.5, args.fr_size_0, endpoint=False)
    xgrid_1 = np.linspace(-0.5, 0.5, args.fr_size_1, endpoint=False)
    if args.kernel_type == 'triangle':
        kernel_param = args.triangle_slope / args.signal_dim
    else:
        kernel_param_0 = args.gaussian_std / args.signal_dim_0
        kernel_param_1 = args.gaussian_std / args.signal_dim_1
        kernel_param = [kernel_param_0, kernel_param_1]

    return load_dataloader_fixed_noise(args.n_validation, signal_dim_0=args.signal_dim_0, signal_dim_1=args.signal_dim_1,max_n_freq=args.max_n_freq,
                                       min_sep=args.min_sep, distance=args.distance, amplitude=args.amplitude,
                                       floor_amplitude=args.floor_amplitude, kernel_type=args.kernel_type,
                                       kernel_param=kernel_param, batch_size=args.batch_size, xgrid_0=xgrid_0,xgrid_1=xgrid_1,
                                       snr=args.snr, noise=args.noise,th=args.th)
