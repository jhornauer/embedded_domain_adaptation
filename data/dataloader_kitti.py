# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import torch.utils.data as data
import data.transforms as transforms
import pickle
from PIL import Image
import h5py


def img_loader(path):
    img = Image.open(path, mode='r')
    np_img = np.asarray(img)
    return np_img


def h5_loader(path):
    h5f = h5py.File(path, "r")
    depth = np.array(h5f['depth'])
    return depth


class MyDataloader(data.Dataset):
    modality_names = ['rgb']

    def find_classes(self, dir):
        all_scenes = []
        for target in sorted(os.listdir(dir)):
            target_dir = os.path.join(dir, target)
            classes = [os.path.join(target, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
            classes.sort()
            all_scenes += classes
        scenes_to_idx = {all_scenes[i]: i for i in range(len(all_scenes))}
        return all_scenes, scenes_to_idx

    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for scene in sorted(os.listdir(dir)):
            for target in os.listdir(os.path.join(dir, scene)):
                d = os.path.join(dir, scene, target)
                if not os.path.isdir(d):
                    continue
                d = os.path.join(d, 'image_02', 'data')
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        path = os.path.join(d, fname)
                        item = (path, class_to_idx[os.path.join(scene, target)])
                        images.append(item)
        return images

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, src_file, transform, modality='rgb', loader=img_loader):
        classes, class_to_idx = self.find_classes(root)
        imgs = self.make_dataset(root, class_to_idx)
        assert len(imgs) > 0, "Found 0 images in subfolders of: " + root + "\n"
        self.root = root

        if transform == 'train':
            self.transform = self.train_transform
        elif transform == 'valid':
            self.transform = self.valid_transform
        else:
            raise NotImplementedError

        f = open(src_file, 'rb')
        self.imgs = pickle.load(f)
        f.close()

        self.loader = loader

        assert (modality in self.modality_names), "Invalid modality split: " + modality + "\n" + \
                                                  "Supported dataset splits are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def valid_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def __getraw__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        path_img = os.path.join(self.root, path)
        # path_img = path_img.replace('jpg', 'png')
        path_depth = path.replace('image_02', 'depth')
        path_depth = path_depth.replace('jpg', 'h5')
        path_depth = os.path.join(self.root, path_depth)
        rgb = self.loader(path_img)
        depth = h5_loader(path_depth)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise (RuntimeError("transform not defined"))

        if self.modality == 'rgb':
            input_np = rgb_np

        to_tensor = transforms.ToTensor()
        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)


