from __future__ import print_function

import random

from utils import get_config
from trainer import UNIT_Trainer
import matplotlib.pyplot as plt
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/unit_noise2clear-bn.yaml', help="net configuration")
parser.add_argument('--input_a', type=str, default = "./dataset/data_a/", help="input image path")
parser.add_argument('--input_c', type=str, default = "./dataset/data_c/", help="input image path")


parser.add_argument('--output_folder', type=str, default='./result_patch', help="output image path")
parser.add_argument('--checkpoint', type=str, default='./outputs/unit_noise2clear-bn/checkpoints/gen_00230000.pt',
                    help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
parser.add_argument('--psnr', action="store_false", help='is used to compare psnr')
parser.add_argument('--ref', type=str, default='J:\\Public_DataSet\\Kodak\\original\\kodim04.png', help='cmpared refferd image')
opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

config = get_config(opts.config)

trainer = UNIT_Trainer(config)
state_dict = torch.load(opts.checkpoint, map_location='cpu')
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.gen_c.load_state_dict(state_dict['c'])

trainer.cuda()
trainer.eval()
encode_a_sty = trainer.gen_a.encode_sty
encode_b = trainer.gen_b.encode_cont
encode_c = trainer.gen_c.encode_cont 
decode_b = trainer.gen_b.dec_cont 
decode_a = trainer.gen_a.dec_recs

if not os.path.exists(opts.input_c):
    raise Exception('input path is not exists!')

stylelist = os.listdir(opts.input_a)
imglist = os.listdir(opts.input_c)

for i, file in enumerate(imglist):
    print(file)
    filepath = opts.input_c + '/' + file
    style_file = random.choice(stylelist)
    stylepath = opts.input_a + '/' + style_file
    image = Image.open(filepath).convert('RGB')
    style_image = Image.open(stylepath).convert('RGB')
    height, width = image.size[0], image.size[1]
    transform = transforms.Compose([transforms.Resize((height, width)),
                                        transforms.ToTensor()])

    origin_path = os.path.join(opts.output_folder, "origin")
    if not os.path.exists(origin_path):
      os.makedirs(origin_path)
    origin_img = os.path.join(origin_path, file)
    image.save(origin_img, quality=95)



    image = transform(image).unsqueeze(0).cuda()

    style_image = transform(style_image).unsqueeze(0).cuda()

    h, w = image.size(2), image.size(3)
    if h > 800 or w > 800:
        continue
    pad_h = h % 4
    pad_w = w % 4
    image = image[:,:,0:h-pad_h, 0:w - pad_w]



    with torch.no_grad():
        content_c = encode_c(image)
        style = encode_a_sty(style_image)
        image_b = decode_b(content_c)
        if not os.path.exists(os.path.join(opts.output_folder,"high_quality")):
            os.makedirs(os.path.join(opts.output_folder, "high_quality"))
        path = os.path.join(opts.output_folder, "high_quality", file)
        vutils.save_image(image_b.data, path, padding=0, normalize=True)

        content_b = encode_b(image_b)
        content = torch.cat([content_b, style], 1)
        outputs = decode_a(content)
        if not os.path.exists(os.path.join(opts.output_folder,"results")):
            os.makedirs(os.path.join(opts.output_folder, "results"))
        path = os.path.join(opts.output_folder, "results", file)
        outputs_back = outputs.clone()
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
    # if opts.psnr:
    #     outputs = torch.squeeze(outputs_back)
    #     outputs = outputs.permute(1, 2, 0).to('cpu', torch.float32).numpy()
    #     # outputs = outputs.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #     ref = Image.open(opts.ref).convert('RGB')
    #     ref = np.array(ref) / 255.
    #     noi = Image.open(opts.input).convert('RGB')
    #     noi = np.array(noi) / 255.
    #     rmpad_h = noi.shape[0] % 4
    #     rmpad_w = noi.shape[1] % 4

    #     pad_h = ref.shape[0] % 4
    #     pad_w = ref.shape[1] % 4

        # if rmpad_h != 0 or pad_h != 0:
        #     noi = noi[0:noi.shape[0]-rmpad_h,:,:]
        #     ref = ref[0:ref.shape[0]-pad_h,:,:]
        # if rmpad_w != 0 or pad_w != 0:
        #     noi = noi[:, 0:noi.shape[1]-rmpad_w,:]
        #     ref = ref[:, 0:ref.shape[1]-pad_w,:]
            
        # psnr = compare_psnr(ref, outputs)
        # ssim = compare_ssim(ref, outputs, multichannel=True)
        # print('psnr:{}, ssim:{}'.format(psnr, ssim))
        # plt.figure('ref')
        # plt.imshow(ref, interpolation='nearest')
        # plt.figure('out')
        # plt.imshow(outputs, interpolation='nearest')
        # plt.figure('in')
        # plt.imshow(noi, interpolation='nearest')
        # plt.show()

    
    

