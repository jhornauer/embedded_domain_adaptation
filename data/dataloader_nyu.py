"""
Soruce: https://github.com/dwofk/fast-depth/blob/master/dataloaders/dataloader.py
"""
import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import data.transforms as transforms
import pickle


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


class MyDataloader(data.Dataset):
    modality_names = ['rgb']

    def is_image_file(self, filename):
        IMG_EXTENSIONS = ['.h5']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

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
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, split, src_file=None, modality='rgb', loader=h5_loader):
        classes, class_to_idx = self.find_classes(root)
        imgs = self.make_dataset(root, class_to_idx)
        assert len(imgs) > 0, "Found 0 images in subfolders of: " + root + "\n"
        self.root = root

        if src_file is None:
            self.imgs = imgs
            self.classes = classes
            self.class_to_idx = class_to_idx
        else:
            f = open(os.path.join(self.root, src_file), 'rb')
            self.imgs = pickle.load(f)
            f.close()

        if split == 'train':
            self.transform = self.train_transform
        elif split == 'holdout':
            self.transform = self.valid_transform
        elif split == 'val':
            self.transform = self.valid_transform
        else:
            raise (RuntimeError("Invalid dataset split: " + split + "\n"
                                                                    "Supported dataset splits are: train, val"))
        self.loader = loader

        assert (modality in self.modality_names), "Invalid modality split: " + modality + "\n" + \
                                                  "Supported dataset splits are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def valid_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def __getraw__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        path = os.path.join(self.root, path)
        rgb, depth = self.loader(path)
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
