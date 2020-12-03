import argparse 
import os 
from network.two_stage_vae_model import *
import numpy as np 
import torch
import math 
import time 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from scipy.misc import imsave, imresize
from fid_score import evaluate_fid_score
import pickle
from dataset import load_dataset, load_test_dataset


def main():
    # exp info
    exp_folder = os.path.join(args.output_path, args.dataset, args.exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    model_path = os.path.join(exp_folder, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # model
    encoder1 = WaeEncoder()
    decoder1 = WaeDecoder()
    encoder1.eval()
    decoder1.eval()
    
    # test dataset 
    x, side_length, channels = load_test_dataset(args.dataset, args.root_folder)
    np.random.shuffle(x)
    x = x[0:10000]
    np.transpose(x, [0, 3, 1, 2])

    # reconstruction and generation
    def reconstruct(x):
        num_sample = np.shape(x)[0]
        mu_z, sd_z, logsd_z, z = encoder1(x)
        x_hat, loggamma_x, gamma_x = decoder1(z)
        return x_hat
    
    def generate(self, num_sample, stage=2):
        if stage == 2:
            u = np.random.normal(0, 1, [num_sample, self.latent_dim])
            z_hat, loggamma_z, gamma_z = decoder2(u)
            z = z_hat + gamma_z * np.random.normal(0, 1, [num_sample, self.latent_dim])
        else:
            z = np.random.normal(0, 1, [num_sample, self.latent_dim])
        x_hat, _, _ = decoder1(z)
        return x_hat
    
    img_recons = reconstruct(x)
    img_gens1 = generate(10000, 1)
    img_gens2 = generate(10000, 2)
    
    np.transpose(img_recons, [0, 3, 1, 2])
    np.transpose(img_gens1, [0, 3, 1, 2])
    np.transpose(img_gens2, [0, 3, 1, 2])

    img_recons_sample = stich_imgs(img_recons)
    img_gens1_sample = stich_imgs(img_gens1)
    img_gens2_sample = stich_imgs(img_gens2)
    plt.imsave(os.path.join(exp_folder, 'recon_sample.jpg'), img_recons_sample)
    plt.imsave(os.path.join(exp_folder, 'gen1_sample.jpg'), img_gens1_sample)
    plt.imsave(os.path.join(exp_folder, 'gen2_sample.jpg'), img_gens2_sample)

    # calculating FID score
    fid_recon = evaluate_fid_score(img_recons.copy(), args.dataset, args.root_folder, True)
    fid_gen1 = evaluate_fid_score(img_gens1.copy(), args.dataset, args.root_folder, True)
    fid_gen2 = evaluate_fid_score(img_gens2.copy(), args.dataset, args.root_folder, True)
    print('Reconstruction Results:')
    print('FID = {:.4F}\n'.format(fid_recon))
    print('Generation Results (First Stage):')
    print('FID = {:.4f}\n'.format(fid_gen1))
    print('Generation Results (Second Stage):')
    print('FID = {:.4f}\n'.format(fid_gen2))


def stich_imgs(x, img_per_row=10, img_per_col=10):
    x_shape = np.shape(x)
    assert(len(x_shape) == 4)
    output = np.zeros([img_per_row*x_shape[1], img_per_col*x_shape[2], x_shape[3]])
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

    args = parser.parse_args()
    print(args)

    main()