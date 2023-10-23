import os 
import numpy as np 
import imageio
import cv2 

def get_image_filenames(dir):
    """Returns all files in the input directory dir that are images"""
    if 'train' in dir: 
        train_file = os.path.join(dir, 'train_list.txt')
        reader = open(train_file, 'r')
        images = []
        for line in reader:
            cur_path = line.strip()
            cur_path = os.path.join(data_root, '/'.join(cur_path.split('/')[-3:]))
            images.append(cur_path)
        
    else:
        image_types = ('exr')
        files = os.listdir(dir) 
        exts = (os.path.splitext(f)[1] for f in files)
        images = [os.path.join(dir, f)
                for e, f in zip(exts, files)
                if e[1:] in image_types]
    return images

def check_image(image_path):
    img_name = image_path 
    depth_name = str.replace(img_name, 'img', 'depth')
    amp_name = str.replace(img_name, 'img', 'amp')
    phase_name = str.replace(img_name, 'img', 'phs')

    # print(img_name, depth_name, phase_name, amp_name)
    im = cv2.imread(img_name, -1)
    amp = cv2.imread(amp_name, -1)
    phase = cv2.imread(phase_name, -1)
    depth = cv2.imread(depth_name, -1)

    if im is not None and amp is not None and phase is not None and depth is not None:
        pass 
    else:
        print('This file can not be successfully read: ', image_path)

data_root = '/mnt/data/home/yujie/yujie_data/MIT_CGH_4K'

for split in ['train', 'validate', 'test']:
    subfolder = os.path.join(data_root, f'{split}_384')
    if split == 'validate' or split == 'test':
        subfolder = os.path.join(subfolder, 'img')
    file_paths = get_image_filenames(subfolder)
    #print(file_paths)

    for img_path in file_paths:
        check_image(img_path)