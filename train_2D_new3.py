import os
import sys
import time
import argparse
import logging

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True
from data import dataset_2D_new3
import modules_2D_new3
import util_2D_3
from data.noise_2D_3 import noise_torch


logger = logging.getLogger(__name__)

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

def train_pre_representation(args, pre_module, pre_optimizer, pre_criterion, pre_scheduler, train_loader, val_loader,
                                   xgrid, epoch, tb_writer):
    """
    Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    pre_module.train()
    loss_train_pre = 0

    for batch_idx, (clean_signal, target_fr, freq,_,fr_cost) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal,target_fr,fr_cost = clean_signal.cuda(),target_fr.cuda(),fr_cost.cuda()
        noisy_signal = noise_torch(clean_signal, args.snr, args.noise)
        for i in range(noisy_signal.size()[0]):
            mv=torch.max(torch.sqrt(pow(noisy_signal[i][0],2)+pow(noisy_signal[i][1],2)))
            noisy_signal[i][0]=noisy_signal[i][0]/mv
            noisy_signal[i][1]=noisy_signal[i][1]/mv
        pre_optimizer.zero_grad()
        output_pre= pre_module(noisy_signal)

        loss_pre = torch.pow((output_pre - target_fr),2)*fr_cost
        loss_pre = torch.sum(loss_pre).to(torch.float32)
        loss_pre.backward()
        pre_optimizer.step()
        loss_train_pre += loss_pre.data.item()


    pre_module.eval()
    loss_val_pre = 0

    for batch_idx, (noisy_signal, clean_signal, target_fr, freq,_,fr_cost) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, target_fr,clean_signal,fr_cost = noisy_signal.cuda(), target_fr.cuda(),clean_signal.cuda(),fr_cost.cuda()
            for i in range(noisy_signal.size()[0]):
                mv=torch.max(torch.sqrt(pow(noisy_signal[i][0],2)+pow(noisy_signal[i][1],2)))
                noisy_signal[i][0]=noisy_signal[i][0]/mv
                noisy_signal[i][1]=noisy_signal[i][1]/mv
        with torch.no_grad():
            output_pre = pre_module(noisy_signal)

        loss_pre = torch.pow((output_pre - target_fr),2)*fr_cost
        loss_pre = torch.sum(loss_pre).to(torch.float32)
        loss_val_pre += loss_pre.data.item()


    loss_train_pre /= args.n_training
    loss_val_pre /= args.n_validation


    tb_writer.add_scalar('train_loss', loss_train_pre, epoch)
    tb_writer.add_scalar('validation_loss', loss_val_pre, epoch)

    pre_scheduler.step(loss_val_pre)
    logger.info("Epochs: %d / %d, Time: %.1f, PRE training L2 loss %.2f, FR validation L2 loss %.2f",
                epoch, args.n_epochs_pre , time.time() - epoch_start_time, loss_train_pre, loss_val_pre)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoint_2D_2/experiment', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used during training')
    parser.add_argument('--signal_dim_0', type=int, default=8, help='dimensionof the input signal')
    parser.add_argument('--signal_dim_1', type=int, default=64, help='dimensionof the input signal')
    parser.add_argument('--fr_size_0', type=int, default=8*8, help='size of the frequency representation')
    parser.add_argument('--fr_size_1', type=int, default=64*8, help='size of the frequency representation')
    parser.add_argument('--max_n_freq', type=int, default=10,
                        help='for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq')
    parser.add_argument('--min_sep', type=float, default=1.,
                        help='minimum separation between spikes, normalized by signal_dim')
    parser.add_argument('--distance', type=str, default='normal', help='distance distribution between spikes')
    parser.add_argument('--amplitude', type=str, default='normal_floor', help='spike amplitude distribution')
    parser.add_argument('--floor_amplitude', type=float, default=0.1, help='minimum amplitude of spikes')
    parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
    parser.add_argument('--snr', type=float, default=-15, help='snr parameter')
    # frequency-representation (fr) module parameters
    parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: [fr | psnet]')
    parser.add_argument('--fr_n_layers', type=int, default=20, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=36, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=5,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=3, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=250, help='dimension after first linear transformation')
    parser.add_argument('--fr_upsampling', type=int, default=2,
                        help='stride of the transposed convolution, upsampling * inner_dim = fr_size')

    # kernel parameters used to generate the ideal frequency representation
    parser.add_argument('--kernel_type', type=str, default='gaussian',
                        help='type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]')
    parser.add_argument('--triangle_slope', type=float, default=8000,
                        help='slope of the triangle kernel normalized by signal_dim')
    parser.add_argument('--gaussian_std', type=float, default=0.12,
                        help='std of the gaussian kernel normalized by signal_dim')
    # training parameters
    parser.add_argument('--n_training', type=int, default=30000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=1000, help='# of validation data')
    parser.add_argument('--lr_fr', type=float, default=0.003,
                        help='initial learning rate for adam optimizer used for the frequency-representation module')
    parser.add_argument('--n_epochs_pre', type=int, default=450, help='number of epochs used to train the fr module')
    parser.add_argument('--save_epoch_freq', type=int, default=10,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=190)
    parser.add_argument('--torch_seed', type=int, default=113)
    parser.add_argument('--th', type=float, default=0.2)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    tb_writer = SummaryWriter(args.output_dir)
    util_2D_3.print_args(logger, args)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)


    train_loader = dataset_2D_new3.make_train_data(args)
    val_loader = dataset_2D_new3.make_eval_data(args)

    pre_module=modules_2D_new3.set_pre_skip_module(args)
    pre_optimizer, pre_scheduler = util_2D_3.set_optim(args, pre_module, 'pre-skip')
    start_epoch = 1

    logger.info('[Network] Number of parameters in the fre-representation module : %.3f M' % (
            util_2D_3.model_parameters(pre_module) / 1e6))

    pre_criterion = torch.nn.MSELoss(reduction='sum')

    xgrid_0 = np.linspace(-0.5, 0.5, args.fr_size_0, endpoint=False)
    xgrid_1 = np.linspace(-0.5, 0.5, args.fr_size_1, endpoint=False)

    for epoch in range(start_epoch, args.n_epochs_pre + 1):
        train_pre_representation(args=args, pre_module=pre_module, pre_optimizer=pre_optimizer, pre_criterion=pre_criterion,
                                           pre_scheduler=pre_scheduler, train_loader=train_loader, val_loader=val_loader,
                                           xgrid=[xgrid_0,xgrid_1], epoch=epoch, tb_writer=tb_writer)



        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs_pre:
            util_2D_3.save(pre_module, pre_optimizer, pre_scheduler, args, epoch, 'pre')

