import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # print(image.shape)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, transform2=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.transform2 = transform2

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # print(image.shape)
            # print(label.shape)
            # if self.transform2:
            #     print("IN")
            #     print(image.shape)
            #     print(label.shape)
            #     print("OUT")
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        # print(sample['image'].shape)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        grayscale = sample['image']
        # grayscale = sample['image'].type('torch.FloatTensor')
        sample['image'] = torch.zeros((3, grayscale.shape[1], grayscale.shape[2]), dtype=torch.float)
        # print(grayscale.shape)
        try:
            sample['image'][0] = grayscale.type('torch.FloatTensor')
            sample['image'][1] = grayscale.type('torch.FloatTensor')
            sample['image'][2] = grayscale.type('torch.FloatTensor')
        except:
            # grayscale = grayscale.reshape((grayscale.shape[0], 1, grayscale.shape[1], grayscale.shape[2]))
            # if grayscale.shape[0] == 192:
            sample['image'] = np.zeros((3, grayscale.shape[0], grayscale.shape[1], grayscale.shape[2]))
            sample['image'][0,:,:,:] = grayscale
            sample['image'][1,:,:,:] = grayscale
            sample['image'][2,:,:,:] = grayscale
            # else:
            #     sample['image'] = np.zeros((grayscale.shape[0], 3, grayscale.shape[1], grayscale.shape[2]))
            #     sample['image'][:,0,:,:] = grayscale
            #     sample['image'][:,1,:,:] = grayscale
            #     sample['image'][:,2,:,:] = grayscale
        # print(type(sample['image'][0,0,0]))
        return sample
