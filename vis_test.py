import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim

def main():
    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()

def vis_mayo(folder, gtfolder):
    filelist = os.listdir(folder)
    count = len(filelist)
    psnr = 0
    ssim = 0
    for file in filelist:
        mat = np.load(folder + '\\' + file)
        gtmat = np.load(gtfolder + '\\' + file)
        mat = np.clip(mat, 0, 1)
        psnr += compare_psnr(gtmat, mat)
        ssim += compare_ssim(gtmat, mat)
        savename = folder + '\\' + file.replace('npy', 'png')
        plt.imsave(savename, mat, vmin=860 /3000, vmax=1260/3000, cmap='gray')
    print('psnr: {}  ssim: {}'.format(psnr / count, ssim/count))

if __name__ == '__main__':
    folder = 'K:\\DAS\\outputs'
    gtfolder = 'K:\\data\\ref'
    vis_mayo(folder, gtfolder)