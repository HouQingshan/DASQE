import os
import PIL
import math
from PIL import Image
import numpy as np
from tqdm import tqdm


def img_crop(image_path, patch_nums):
    input = Image.open(image_path)
    w, h = input.size
    crop_num = int(math.sqrt(patch_nums))
    crop_width = int(math.ceil( w / crop_num))
    crop_height = int(math.ceil( h / crop_num))

    patches = []

    start_w = 0
    start_h = 0
    for i in range(crop_num):
        for j in range(crop_num):
            patch = input.crop((start_w, start_h, min(start_w + crop_width, w), min(start_h + crop_height, h)))
            patches.append(patch)
            start_w += crop_width
        start_w = 0
        start_h += crop_height

    # output patches path
    save_path = "./patches/{}".format(image_path.split("/")[-1].split(".")[0])
    all_path = "./allpatchs"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(all_path):
        os.makedirs(all_path)

    for idx in range(patch_nums):
        name = image_path.split("/")[-1].split(".")[0]
        patches[idx].save(os.path.join(save_path, "{}_patch_{:03d}.png".format(name, idx)))
        patches[idx].save(os.path.join(all_path, "{}_patch_{:03d}.png".format(name, idx)))

    return patches


def img_compose(patch_path):
    img_list = os.listdir(patch_path)
    patchs = []
    for file in img_list:
        patchs.append(Image.open(os.path.join(patch_path, file)))
    h, w = (0, 0)

    num_patchs = int(math.sqrt(len(patchs)))
    for i in range(num_patchs):
        w += patchs[i].size[0]

    for j in range(num_patchs):
        h += patchs[j * num_patchs].size[1]

    res = Image.new("RGB", (w, h))

    start_w = 0
    start_h = 0
    for i in range(num_patchs):
        for j in range(num_patchs):
            res.paste(patchs[i * num_patchs + j], (start_w, start_h))
            start_w += patchs[i * num_patchs + j].size[0]
        start_w = 0
        start_h += patchs[i * num_patchs].size[1]
        
    save_path = "./cover"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    name = patch_path.split("/")[-1]
    res.save(os.path.join(save_path, "{}.png".format(name)))
    return res

def img_check(patchs, save_path):
    def check(patchs, img):
        image = Image.open(os.path.join(patchs, img))
        w, h = image.size
        image = np.array(image)
        num = 0
        left_down = (image[0][0] > 30).any()
        if left_down == False:
            num+=1
        right_down = (image[w-1][0] > 30).any()
        if right_down == False:
            num+=1
        left_up = (image[0][h-1] > 30).any()
        if left_up == False:
            num+=1
        right_up = (image[w-1][h-1] > 30).any()
        if right_up == False:
            num+=1
        center = (image[int(w/2)][int(h/2)] > 30).any()
        if center == False:
            num += 1
            
        if num>1:
            return False
        else:
            return True
        # return (left_down or right_down or left_up or right_up)
`
    res = []
    for patch in tqdm(os.listdir(patchs)):
        if check(patchs, patch):
            res.append(patch)
    
    print("Accepted:{}".format(len(res)))
    
    for i in tqdm(res):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image = Image.open(os.path.join(patchs, i))
        image.save(os.path.join(save_path, "{}".format(i)))
    
    return res


# original image path
filename = 'train/' 
imgsname = os.listdir(filename)

# 1. crop into patches
for i in imgsname:
   img_path= filename+i
   res = img_crop(img_path, 16)

# 2. remove patches with large black edges
composedpath = "./allpatchs/"   # source path
save_path = "./filtered/"       # target path

res_no_black = img_check(composedpath, save_path)    



