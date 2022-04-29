# -*- coding: utf-8 -*-
import numpy as np
import data.transforms as transforms
from data.dataloader_dimlrgbd import MyDataloader

iheight, iwidth = 756, 1344  # raw image size


class DIMLRGBDDataset(MyDataloader):
    def __init__(self, root, split, src_file=None, modality='rgb'):
        self.split = split
        super(DIMLRGBDDataset, self).__init__(root, split, src_file, modality)
        self.output_size = (224, 224)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop((228, 304)),
            transforms.HorizontalFlip(do_flip),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = depth_np * 1e-3  # scale [mm] to [m]
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def valid_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = depth_np * 1e-3  # scale [mm] to [m]
        depth_np = transform(depth_np)

        return rgb_np, depth_np


