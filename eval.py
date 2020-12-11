import matplotlib.pyplot as plt
from model.stage1 import Wae
from model.stage1 import InfoGAN
from model.stage1 import Resnet
from model.stage2.s2vae import *
from dataset import load_dataset, load_test_dataset
import pickle
from fid_score import evaluate_fid_score
from scipy.misc import imsave, imresize
import argparse
import os
import numpy as np
import tensorflow as tf 
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
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    # test dataset
    x, side_length, channels = load_test_dataset(
        args.dataset, args.root_folder)
    np.random.shuffle(x)
    x = x[0:10000] / 255.
    x = np.transpose(x, [0, 3, 1, 2])
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

    encoder1.eval()
    decoder1.eval()
    encoder1.to(device)
    decoder1.to(device)
    
    encoder2 = S2Encoder(args.latent_dim, args.latent_dim,
                         args.second_dim, args.second_depth, args.batch_size)
    decoder2 = S2Decoder(args.latent_dim, args.latent_dim,
                         args.second_dim, args.second_depth, args.batch_size)
    encoder2.eval()
    decoder2.eval()
    encoder2.to(device)
    decoder2.to(device)

    filename = "s1_latest"
    model_dict = torch.load(os.path.join(args.model_path, filename + ".tar"))
    encoder1.load_state_dict(model_dict["encoder"])
    decoder1.load_state_dict(model_dict["decoder"])

    filename = "s2_latest"
    model_dict = torch.load(os.path.join(args.model_path, filename + ".tar"))
    encoder2.load_state_dict(model_dict["encoder"])
    decoder2.load_state_dict(model_dict["decoder"])


    # reconstruction and generation
    def reconstruct(x):
        num_sample = np.shape(x)[0]
        num_iter = math.ceil(float(num_sample) / float(args.batch_size))
        x_extend = np.concatenate([x, x[0:args.batch_size]], 0)
        recon_x = []
        for i in range(num_iter):
            batch = torch.from_numpy(x_extend[i*args.batch_size:(i+1)*args.batch_size]).type(torch.float32).to(device)
            mu_z, sd_z, logsd_z, z = encoder1(batch)
            x_hat, loggamma_x, gamma_x = decoder1(z)
            recon_x.append(x_hat.cpu().detach().numpy())
        recon_x = np.concatenate(recon_x, 0)[0:num_sample]
        return recon_x

    def generate(num_sample, stage=2):
        num_iter = math.ceil(float(num_sample) / float(args.batch_size))
        gen_samples = []
        for i in range(num_iter):
            if stage == 2:
                # u ~ N(0, I)
                u = torch.randn([args.batch_size, args.latent_dim]).to(device)
                # z ~ N(f_2(u), \gamma_z I)
                z_hat, loggamma_z, gamma_z = decoder2(u)
                z = z_hat + gamma_z * torch.randn([args.batch_size, args.latent_dim]).to(device)
            else:
                z = torch.randn([args.batch_size, args.latent_dim]).to(device)
            # x = f_1(z)
            x_hat, _, _ = decoder1(z)
            gen_samples.append(x_hat.cpu().detach().numpy())
        gen_samples = np.concatenate(gen_samples, 0)
        return gen_samples[0:num_sample]

    img_recons = reconstruct(x)
    img_gens1 = generate(10000, 1)
    img_gens2 = generate(10000, 2)

    img_recons = np.transpose(img_recons, [0, 2, 3, 1])
    img_gens1 = np.transpose(img_gens1, [0, 2, 3, 1])
    img_gens2 = np.transpose(img_gens2, [0, 2, 3, 1])
    
    

    img_recons_sample = stich_imgs(img_recons)
    img_gens1_sample = stich_imgs(img_gens1)
    img_gens2_sample = stich_imgs(img_gens2)
    plt.imsave(os.path.join(exp_folder, 'recon_sample.jpg'), img_recons_sample)
    plt.imsave(os.path.join(exp_folder, 'gen1_sample.jpg'), img_gens1_sample)
    plt.imsave(os.path.join(exp_folder, 'gen2_sample.jpg'), img_gens2_sample)

    # calculating FID score
    fid_score_log = 
    
    fid_recon = evaluate_fid_score(
        img_recons.copy(), args.dataset, args.root_folder, True)
    fid_gen1 = evaluate_fid_score(
        img_gens1.copy(), args.dataset, args.root_folder, True)
    fid_gen2 = evaluate_fid_score(
        img_gens2.copy(), args.dataset, args.root_folder, True)
    print('Reconstruction Results:')
    print('FID = {:.4F}\n'.format(fid_recon))
    print('Generation Results (First Stage):')
    print('FID = {:.4f}\n'.format(fid_gen1))
    print('Generation Results (Second Stage):')
    print('FID = {:.4f}\n'.format(fid_gen2))
    
    with open()


def stich_imgs(x, img_per_row=10, img_per_col=10):
    x_shape = np.shape(x)
    assert(len(x_shape) == 4)
    output = np.zeros(
        [img_per_row*x_shape[1], img_per_col*x_shape[2], x_shape[3]])
    idx = 0
    for r in range(img_per_row):
        start_row = r * x_shape[1]
        end_row = start_row + x_shape[1]
        for c in range(img_per_col):
            start_col = c * x_shape[2]
            end_col = start_col + x_shape[2]
            output[start_row:end_row, start_col:end_col] = x[idx]
            idx += 1
            if idx == x_shape[0]:
                break
        if idx == x_shape[0]:
            break
    if np.shape(output)[-1] == 1:
        output = np.reshape(output, np.shape(output)[0:2])
    return output


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
    parser.add_argument('--cross-entropy-loss',
                        default=False, action='store_true')

    args = parser.parse_args()
    print(args)
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpus)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[args.gpu],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
    )

    main()
