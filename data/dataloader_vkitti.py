# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import torch.utils.data as data
import data.transforms as transforms
import pickle
from PIL import Image


def img_loader(path):
    img = Image.open(path, mode='r')
    np_img = np.array(img)
    return np_img


class MyDataloader(data.Dataset):
    modality_names = ['rgb']

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for modality in sorted(os.listdir(d)):
                md = os.path.join(d, modality)
                if not os.path.isdir(md):
                    continue
                md = os.path.join(md, 'frames', 'rgb', 'Camera_0')
                for root, _, fnames in sorted(os.walk(md)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, src_file, transform, modality='rgb', loader=img_loader):
        classes, class_to_idx = self.find_classes(os.path.join(root, 'rgb'))
        imgs = self.make_dataset(os.path.join(root, 'rgb'), class_to_idx)
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
        path_depth = path.replace('rgb', 'depth')
        path_depth = path_depth.replace('jpg', 'png')
        path_depth = os.path.join(self.root, path_depth)
        rgb = self.loader(path_img)
        depth = self.loader(path_depth)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise (RuntimeError("transform not defined"))

        normalize_rgb = transforms.NormalizeNumpyArray((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        rgb_np = normalize_rgb(rgb_np)

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


