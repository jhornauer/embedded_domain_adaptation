# -*- coding: utf-8 -*-
import numpy as np
import data.transforms as transforms
from data.dataloader_kitti import MyDataloader

iheight, iwidth = 375, 1242  # raw image size


class KITTIDataset(MyDataloader):
    def __init__(self, root, src_file, transform, output_size=(256, 512), modality='rgb'):
        self.src_file = src_file
        self.output_size = output_size
        super(KITTIDataset, self).__init__(root, src_file, transform, modality)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        transform = transforms.Compose([
            transforms.Resize(300.0 / rgb.shape[0]),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop((300, 728)),
            transforms.HorizontalFlip(do_flip),
            transforms.Resize(self.output_size),
        ])

        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255

        depth_np = transform(depth_np)
        depth_np = depth_np * 1e2  # m to cm
        depth_np[depth_np > 8000] = 8000
        depth_np = depth_np / 8000
        return rgb_np, depth_np

    def valid_transform(self, rgb, depth):
        transform = transforms.Compose([
            transforms.Resize(300.0 / rgb.shape[0]),
            transforms.CenterCrop((300, 728)),
            transforms.Resize(self.output_size)
        ])

        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255

        depth_np = transform(depth)
        # depth is given in meters --> no changes needed
        return rgb_np, depth_np
