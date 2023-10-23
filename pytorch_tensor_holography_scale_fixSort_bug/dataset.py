from torch.utils.data import Dataset, DataLoader
import torch
import glob
import numpy as np
import os
import json
import h5py
import cv2
import random
from matplotlib import pyplot as plt
from scipy import io as sio
from matplotlib.pyplot import MultipleLocator
import random 
from propagation_utils import gen_mask 

from imageio import imread
from skimage.transform import resize
import utils.utils as utils
import time


def get_dataloader(phase, config, num_workers=4):
    is_shuffle = phase == 'train'
    channels = {'r':0, 'g':1, 'b':2, 'rgb':None}
    image_res = (config.img_res, config.img_res)
    roi_res = (config.img_res, config.img_res)
    if phase != 'train':
        data_path = os.path.join(config.holo_data_root, f'{phase}_{config.img_res}/img')
    else: 
         data_path = os.path.join(config.holo_data_root, f'{phase}_{config.img_res}')

    dataloader = ImageLoader(data_path, channel=channels[config.channel],
                           image_res=image_res, homography_res=roi_res, batch_size=config.batch_size,
                           crop_to_homography=True, shuffle=is_shuffle, n_fixed_depth=config.n_fixed_depth,
                           n_float_depth = config.n_float_depth)

    return dataloader

# Classes
##########################################################
class ImageLoader:
    """Loads images a folder with augmentation for generator training

    Class initialization parameters
    -------------------------------
    data_path: folder containing images
    channel: color channel to load (0, 1, 2 for R, G, B, None for all 3),
        default None
    batch_size: number of images to pass each iteration, default 1
    image_res: 2d dimensions to pad/crop the image to for final output, default
        (1080, 1920)
    homography_res: 2d dims to scale the image to before final crop to image_res
        for consistent resolutions (crops to preserve input aspect ratio),
        default (880, 1600)
    shuffle: True to randomize image order across batches, default True
    crop_to_homography: if True, only crops the image instead of scaling to get
        to target homography resolution, default False
    """

    def __init__(self, data_path, channel=None, batch_size=1,
                 image_res=(384, 384), homography_res=(384, 384), shuffle=True,
                 idx_subset=None, crop_to_homography=False, n_fixed_depth=15, n_float_depth=5):
        print(data_path)
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')
        self.data_path = data_path
        self.channel = channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.homography_res = homography_res
        self.subset = idx_subset
        self.crop_to_homography = crop_to_homography
        self.near_plane = -3
        self.volume_width = 6
        self.n_fixed_depth = n_fixed_depth
        self.n_float_depth = n_float_depth
        self.augmentations = []
        # store the possible states for enumerating augmentations
        self.augmentation_states = [fn() for fn in self.augmentations]

        self.im_names = get_image_filenames(data_path)
        #self.im_names = self.im_names[:2]
        self.im_names.sort()
        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        self.order = list(self.order)

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __next__(self):
        if self.subset is not None:
            while self.ind not in self.subset and self.ind < len(self.order):
                self.ind += 1

        if self.ind < len(self.order):
            batch_ims = self.order[self.ind:self.ind+self.batch_size]
            self.ind += self.batch_size
            return self.load_batch(batch_ims)
        else:
            raise StopIteration

    def __len__(self):
        if self.subset is None:
            return len(self.order)
        else:
            return len(self.subset)

    def load_batch(self, images):
        im_res_name = [self.load_image(*im_data) for im_data in images]
        ims = torch.stack([item[0] for item in im_res_name], 0)
        masks = torch.stack([item[1] for item in im_res_name], 0)
        depths = torch.stack([item[2] for item in im_res_name], 0)
        amps = torch.stack([item[3] for item in im_res_name], 0)
        phs = torch.stack([item[4] for item in im_res_name], 0)
        recon_dists = torch.stack([item[5] for item in im_res_name], 0)
        names = [item[6] for item in im_res_name]
        return ims, masks, depths, amps, phs, recon_dists, names

    def load_image(self, filenum, *augmentation_states):

        img_name = self.im_names[filenum]    
        depth_name = str.replace(img_name, 'img', 'depth')
        amp_name = str.replace(img_name, 'img', 'amp')
        phase_name = str.replace(img_name, 'img', 'phs')

        # print(img_name, depth_name, phase_name, amp_name)
        im = cv2.imread(self.im_names[filenum], -1)
        amp = cv2.imread(amp_name, -1)
        phase = cv2.imread(phase_name, -1)
        depth = cv2.imread(depth_name, -1)
        if len(depth.shape) == 3:
            depth = depth[:,:,0]
        # print(np.max(depth), np.min(depth))
        # print(im.shape, amp.shape, depth.shape, phase.shape)

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))
        amp = np.transpose(amp, axes=(2, 0, 1))

        # compute_depth histogram bins 
        depth_scaled = depth * self.volume_width + self.near_plane
        hist, bins = np.histogram(depth_scaled.ravel(), 200, [self.near_plane, self.near_plane + self.volume_width])
        concate_bins_count = np.stack((bins[:-1], bins[1:], hist), axis=1)
        concate_bins_count = concate_bins_count[np.argsort(concate_bins_count[:,2]),...]
        fixed_bins = concate_bins_count[-self.n_fixed_depth:,:2]
        random_row_indices = random.sample(list(range(185)), 5)
        recon_depth_list = []
        for j in range(self.n_fixed_depth + self.n_float_depth):
            if j < self.n_fixed_depth:
                sample_dist = np.random.random_sample() * (fixed_bins[j,1] - fixed_bins[j,0]) + fixed_bins[j,0]
            else: 
                sample_dist = np.random.random_sample() * (concate_bins_count[random_row_indices[j-self.n_fixed_depth], 1]
                            - concate_bins_count[random_row_indices[j-self.n_fixed_depth], 0]) +  concate_bins_count[random_row_indices[j-self.n_fixed_depth], 0]
            recon_depth_list.append(sample_dist)
        recon_depth_array = np.array(recon_depth_list)
        recon_dists = torch.from_numpy(recon_depth_array).float()

        depth = depth[np.newaxis, ...]  # (h,w) -> (1, h, w)
        phase = np.transpose(phase ,axes=(2,0,1)) 
        phase = ( phase * 2 - 1 ) * np.pi # transform from [0,1] to [-pi, pi]
        

        # normalize resolution
        input_res = im.shape[-2:]
        mask = np.ones(im.shape)
        if self.crop_to_homography:
            im = pad_crop_to_res(im, self.homography_res)
            mask = pad_crop_to_res(mask, self.homography_res)
            depth = pad_crop_to_res(depth, self.homography_res)
            phase = pad_crop_to_res(phase, self.homography_res)
            amp = pad_crop_to_res(amp, self.homography_res)

        im = pad_crop_to_res(im, self.image_res)
        mask = pad_crop_to_res(mask, self.image_res)
        depth = pad_crop_to_res(depth, self.image_res)
        phase = pad_crop_to_res(phase, self.image_res)
        amp = pad_crop_to_res(amp, self.image_res)
        
        im_t = torch.from_numpy(im).float()
        mask = torch.from_numpy(mask).float()
        depth = torch.from_numpy(depth).float()
        phase = torch.from_numpy(phase).float()
        amp = torch.from_numpy(amp).float() 
        
        return im_t, mask, depth, amp, phase, recon_dists, img_name

    def augment_vert(self, image=None, flip=False):
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1, :]
        return image

    def augment_horz(self, image=None, flip=False):
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1]
        return image


def get_image_filenames(dir):
    """Returns all files in the input directory dir that are images"""
    if 'train' in dir: 
        train_file = os.path.join(dir, 'train_list.txt')
        reader = open(train_file, 'r')
        images = []
        for line in reader:
            images.append(line.strip())
    else:
        image_types = ('exr')
        files = os.listdir(dir) 
        exts = (os.path.splitext(f)[1] for f in files)
        images = [os.path.join(dir, f)
                for e, f in zip(exts, files)
                if e[1:] in image_types]
    return images


def resize_keep_aspect(image, target_res, pad=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False)

    # switch to numpy channel dim convention, resize, switch back
    image = np.transpose(image, axes=(1, 2, 0))
    image = resize(image, target_res, mode='reflect')
    return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=False),
                            target_res, pytorch=False)


class HoloDataset(Dataset):
    def __init__(self, phase, img_res, data_root, constant_interval=False):

        data_dir = data_root + f'/{phase}_{img_res:d}/img'
        files = os.listdir(data_dir)
        self.files = [os.path.join(data_dir, item) for item in files]
        self.size = len(self.files)
        
    def __getitem__(self, index):
        image_name = self.files[index]
        depth_name = str.replace(image_name, 'img', 'depth')
        amp_name = str.replace(image_name, 'img', 'amp')
        phase_name = str.replace(image_name, 'img', 'phs')

        img = cv2.imread(image_name, -1)
        depth = cv2.imread(depth_name, -1)
        amp = cv2.imread(amp_name, -1)
        phase = cv2.imread(phase_name, -1)

        height, width, channel = img.shape
        img = img.transpose((2,0,1))
        amp = amp.transpose((2,0,1))
        depth = depth[np.newaxis, ...]
        phase = phase.transpose((2,0,1))

        img = torch.from_numpy(img).float()
        amp = torch.from_numpy(amp).float()
        depth = torch.from_numpy(depth).float()
        phase = torch.from_numpy(phase).float() 
        phase = (phase * 2 - 1) * np.pi # from [0,1] to [-pi, pi]

        return img, depth, amp, phase

    def __len__(self):
        return self.size


if __name__ == "__main__":

    class config:
        def __init__(self, data_root=None):
           
            self.holo_data_root = '/mnt/mnt2/codes/yujie_codes/holonet_compression/3d_hologram_compression/dataset_preparation/MIT_CGH_4K'
            self.batch_size=2
            self.channel = 'rgb'
            self.img_res = 384
            self.n_fixed_depth = 13
            self.n_float_depth = 3
    
    train_config = config()
    dataloader = get_dataloader("test", train_config)
    for k, target in enumerate(dataloader):
    # get target image
        input_img, mask, input_depth, target_amp, target_phase, recon_dists, paths = target
        if k == 0:
            import skimage.io as sio
            target_amp = target_amp[0].numpy().squeeze().transpose((1,2,0))
            target_phase = target_phase[0].numpy().squeeze().transpose((1,2,0))
            mask = mask[0].numpy().squeeze().transpose((1,2,0))
            input_img = input_img[0,...].squeeze().numpy().transpose((1,2,0))
            input_depth = input_depth[0,...].squeeze().numpy()
            print(input_depth.shape)
            print(target_amp.shape)
            print('recon dist tensor \' shape is ', recon_dists.shape)
            print(recon_dists)
            # since we use opencv to read the images, the channel order is BGR
            sio.imsave('test_amp_save.png',target_amp[:,:,::-1])
            sio.imsave('test_mask_save.png', mask[:,:,::-1])
            sio.imsave('test_depth_save.png', input_depth)
            sio.imsave('test_input_save.png', input_img[:,:,::-1])
            sio.imsave('test_phase_save.png', target_phase[:,:,::-1])