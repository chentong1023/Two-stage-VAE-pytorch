import matplotlib.pyplot as plt
from model.stage1 import Wae
from model.stage1 import Resnet
from model.stage1 import InfoGAN
from model.stage2.s2vae import *
from utils.utils_ import save_logfile
from utils.plot_script import plot_loss
from VaeTrainer import VaeTrainer
from torch.utils.data import DataLoader
from dataset import load_dataset, load_test_dataset
import argparse
import os
import numpy as np
import torch
import math
import time
import matplotlib
matplotlib.use('Agg')


def main():
    # exp info
    exp_folder = os.path.join(args.output_path, args.dataset, args.exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    model_path = os.path.join(exp_folder, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    args.model_path = model_path

    # dataset
    x, side_length, channels = load_dataset(args.dataset, args.root_folder)
    num_sample = np.shape(x)[0]
    x = np.transpose(x, [0, 3, 1, 2]) / 255.

    print('Num Sample = {}.'.format(num_sample))

    sampler_x = DataLoader(
        x,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=2,
        shuffle=True
    )
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    print(device)

    # model
    if args.network_structure == 'Resnet':
        encoder1 = Resnet.ResnetEncoder(x.shape, args.num_scale, args.block_per_scale, args.depth_per_block, args.kernel_size, args.base_dim, args.fc_dim, args.latent_dim, device)
        decoder1 = Resnet.ResnetDecoder(x.shape, args.num_scale, args.block_per_scale, args.depth_per_block, args.kernel_size, args.base_dim, args.fc_dim, args.latent_dim)
    elif args.network_structure == 'Infogan':
        encoder1 = InfoGAN.InfoGANEncoder(side_length, side_length, channels, args.latent_dim, args.batch_size)
        decoder1 = InfoGAN.InfoGANDecoder(side_length, side_length, channels, args.latent_dim, args.batch_size)
    elif args.network_structure == 'Wae':
        encoder1 = Wae.WaeEncoder(args.latent_dim, side_length, channels)
        decoder1 = Wae.WaeDecoder(args.latent_dim, side_length, channels)
    else:
        raise Exception("Fuck")

    trainer1 = VaeTrainer(args, sampler_x, device, 1)

    logs1 = trainer1.trainIters(encoder1, decoder1)
    
    print("Stage 1 VAE training has done!")

    plot_loss(logs1, os.path.join(
        exp_folder, "loss_curve_1.png"), args.plot_every)
    save_logfile(logs1, os.path.join(exp_folder, "log_stage_1.txt"))

    #####################################################
    encoder1.eval()
    encoder1.to(device)
    encoder2 = S2Encoder(args.latent_dim, args.latent_dim,
                        args.second_dim, args.second_depth, args.batch_size)
    decoder2 = S2Decoder(args.latent_dim, args.latent_dim,
                        args.second_dim, args.second_depth, args.batch_size)

    trainer2 = VaeTrainer(args, sampler_x, device, 2, True, encoder1)

    logs2 = trainer2.trainIters(encoder2, decoder2)

    plot_loss(logs2, os.path.join(
        exp_folder, "loss_curve_2.png"), args.plot_every)
    save_logfile(logs2, os.path.join(exp_folder, "log_stage_2.txt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', type=str, default='.')
    parser.add_argument('--output-path', type=str, default='./experiments')
    parser.add_argument('--exp-name', type=str, default='debug')

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--network-structure', type=str, default='Infogan')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--write-iteration', type=int, default=600)
    parser.add_argument('--latent-dim', type=int, default=64)

    parser.add_argument('--second-dim', type=int, default=1024)
    parser.add_argument('--second-depth', type=int, default=3)

    parser.add_argument('--num-scale', type=int, default=4)
    parser.add_argument('--block-per-scale', type=int, default=1)
    parser.add_argument('--depth-per-block', type=int, default=2)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--base-dim', type=int, default=16)
    parser.add_argument('--fc-dim', type=int, default=512)

    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr-epochs', type=int, default=150)
    parser.add_argument('--lr-fac', type=float, default=0.5)

    parser.add_argument('--epochs2', type=int, default=800)
    parser.add_argument('--lr2', type=float, default=0.0001)
    parser.add_argument('--lr-epochs2', type=int, default=300)
    parser.add_argument('--lr-fac2', type=float, default=0.5)
    parser.add_argument('--cross-entropy-loss',
                        default=False, action='store_true')

    parser.add_argument('--plot_every', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--save_latest', type=int, default=50)

    parser.add_argument('--is_continue', action="store_true", default=False)
    parser.add_argument('--is_train', action="store_true", default=True)
    
    parser.add_argument('--use_shuffled_vae', action="store_true", default=False)
    parser.add_argument('--alpha_gan', type=float, default=)

    args = parser.parse_args()
    print(args)

    main()
