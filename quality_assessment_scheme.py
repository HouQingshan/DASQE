import os
import random
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as get_psnr

# Step 1 Calculate average PSNR between patches within H 
path = './high'
num_comparisons = 0
total_psnr = 0
img_files = os.listdir(path)
for i in tqdm(range(len(img_files)), desc='CALCULATE AVERAGE PSNR'):
    img1 = np.array(Image.open(os.path.join('high', img_files[i])).convert('L'))
    for j in range(i + 1, len(img_files)):
        img2 = np.array(Image.open(os.path.join('high', img_files[j])).convert('L'))
        total_psnr += get_psnr(img1, img2)
        num_comparisons += 1
AVG_PSNR = total_psnr / num_comparisons
print('average PSNR is: ', AVG_PSNR)



# Step 2 Obtain multiple reference patches
reference_image_number = 500   # define the number of reference patches
reference_images = []
High_list = []
Low_list = []

def calculate_avg_psnr(image, image_list):
    """
    Calculate average psnr between a patch and another patch list
    """ 
    total_psnr = 0
    num_images = len(image_list)
    
    for img in image_list:
        total_psnr += get_psnr(image, img)
        
    return total_psnr / num_images


print('Selecting Reference Patches...')

while len(reference_images) < reference_image_number:
    hi_path = random.choice(os.listdir(path))
    hj_path = random.choice(os.listdir(path))
    
    if hi_path == hj_path or hi_path in reference_images or hi_path in reference_images:
        continue

    hi = io.imread(os.path.join(path, hi_path))
    hj = io.imread(os.path.join(path, hj_path))

    psnr = get_psnr(hi, hj)

    if psnr < AVG_PSNR:
        remaining_images = [image for image in os.listdir(path) if image not in [hi_path, hj_path]]
        remaining_images = [io.imread(os.path.join(path, img)) for img in remaining_images]

        avg_psnr_hi = calculate_avg_psnr(hi, remaining_images)
        avg_psnr_hj = calculate_avg_psnr(hj, remaining_images)

        if avg_psnr_hi > avg_psnr_hj:
            reference_images.append(hi_path)
        else:
            reference_images.append(hj_path)
        print(len(reference_images), '/',  reference_image_number)

reference_image_values = [io.imread(os.path.join(path, img)) for img in reference_images]

print('{} reference patches have been selected.'.format(len(reference_images)))


# Step 3 Updating the high-/low-quality domain according to reference patches
print('Updating the high-/low-quality domain according to reference patches...')
for image_path in tqdm(os.listdir(path)):

    if image_path in reference_images:
        continue

    image = io.imread(os.path.join(path, image_path))
    avg_psnr = calculate_avg_psnr(image, reference_image_values)

    if avg_psnr > AVG_PSNR:
        High_list.append(image_path)
    else:
        Low_list.append(image_path)

print("High_list:", len(High_list))
print("Low_list:", len(Low_list))

print('Copy images...')
os.makedirs('./final_high', exist_ok=True)
os.makedirs('./final_low', exist_ok=True)
for img in High_list:
    shutil.copy(f'./{path}/{img}', 'final_high')
for img in Low_list:
    shutil.copy(f'./{path}/{img}', 'final_low')
print('Done!')