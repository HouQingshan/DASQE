from utils import vgg_preprocess
import torchvision
import torch
import torch.nn as nn
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as image
import scipy.io as sio
from PIL import Image as Image
import numpy as np
import os
class vgg_19(nn.Module):
    def __init__(self):
        super(vgg_19, self).__init__()
        self.vgg_model = torchvision.models.vgg19(pretrained=True)
        self.layers = [6,11,20, 29]
        self.vgg_model.eval()
    def forward(self, x):
        outfea = []
        for layer in self.layers:
            self.feature_ext = nn.Sequential(*list(self.vgg_model.features.children())[:layer])
            out = self.feature_ext(x)[0,:,:,:]
            out = out.mean(dim=0)
            outfea.append(out)
        return outfea

class Statistic_error(object):
    def __init__(self, isGPU):
        super(Statistic_error, self).__init__()        
        self.model = vgg_19()
        self.process = vgg_preprocess
        if isGPU:
            self.model.cuda()

    def forward(self, input, target):
        vgginput = self.process(input)
        vggtarget = self.process(target)
        input_fea = self.model(vgginput)
        target_fea = self.model(vggtarget)
        errors = []
        for index, fea in enumerate(target_fea):
            out1 = fea.detach().numpy()
            out2 = input_fea[index].detach().numpy()
            plt.figure('1')
            plt.imshow(out1, cmap='hot_r')
            plt.figure('2')
            plt.imshow(out2 + out1, cmap='hot_r')
            plt.show()
          
            error = torch.mean((fea - input_fea[index]) ** 2)
            errors.append(error)
        return errors
def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256. 
    depth = np.expand_dims(depth, -1)
    return depth

def image_read(filename):
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=np.uint8)
    img_file.close()
    depth = depth_png.astype(np.float) / 255. 
    depth = np.expand_dims(depth, -1)
    return depth