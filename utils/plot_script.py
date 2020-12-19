import matplotlib.pyplot as plt
import math
import numpy as np
import os
import torch

def plot_loss(losses, save_path, intervals=500):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    for key in losses.keys():
        plt.plot(list_cut_average(losses[key], intervals), label=key)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_factor_variation(encoder1, decoder1, encoder2, decoder2, pic, var_range, save_path, stage):
    print('Start printing factor variation of stage %d VAE.' % stage)
    latent_dim = decoder1.latent_dim
    encoder1.eval()
    decoder1.eval()
    encoder2.eval()
    decoder2.eval()
    if stage == 1:
        pic = torch.tensor(np.array([pic]), dtype=torch.float32)
        _, _, _, pic_latent = encoder1(pic)
    else:
        assert stage == 2
        u = torch.tensor(np.random.normal(0, 1, [1, latent_dim]), dtype=torch.float32)
        pic_latent, _, _ = decoder2(u)
    ran = np.linspace(-var_range, var_range, 50)
    for i in range(latent_dim):
        print("Ploting the %d-th dimension variation" % i)
        plt.figure(figsize=(10, 5))
        plt.title("Variation on the %d-th dim of latent variable" % i)
        for idx, val in enumerate(ran):
            new_pic_latent = pic_latent
            new_pic_latent[0][idx] = val
            new_pic, _, _ = decoder1(new_pic_latent)
            plt.subplot(5, 10, idx + 1)
            plt.axis('off')
            plt.imshow(new_pic[0].T.detach().numpy())
        plt.sag(os.path.join(save_path, ("%d.png" % i)))
        plt.show()