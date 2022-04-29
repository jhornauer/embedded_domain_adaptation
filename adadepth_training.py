import os
import json
import argparse
import torch
from networks.fastdepth import MobileNetSkipAdd, ResNet
from data.vkitti import vKITTIDataset
from data.kitti import KITTIDataset
from data.nyu import NYUDataset
from data.dimlrgbd import DIMLRGBDDataset
from utils import set_requires_grad, weights_init_normal, adadepth_training, test_depth_model
from networks.networks import Discriminator_Latent_Space_KITTI, Discriminator_Latent_Space_DIML, PatchGAN_Discriminator
from utils import merge_dicts, dict2clsattr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='./configs/diml_resnet_1000.json')
    parser.add_argument('-device', type=str, default='cuda:0')
    parser.add_argument('-n_rounds', type=int, default=5)
    parser.add_argument('-reg_val', type=float, default=0.7)
    parser.add_argument('-data_dir', type=str)
    parser.add_argument('-paper', action='store_true', help='loads the paper results if set to true')
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            model_config = json.load(f)
        train_config = vars(args)
    else:
        raise NotImplementedError

    config_dict = merge_dicts(model_config, train_config)
    config = dict2clsattr(config_dict)

    # check if data directory is valid
    if not os.path.isdir(config.data_dir):
        raise NotADirectoryError(config.data_dir)

    if config.dataset == 'kitti':
        if config.resolution == 'low':
            height, width = 256, 512
        elif config.resolution == 'high':
            height, width = 288, 704
        else:
            raise NotImplementedError
    elif config.dataset == 'diml':
        height, width = 224, 224
    else:
        raise NotImplementedError

    patch_height = height // 2 ** 4
    patch_width = width // 2 ** 4

    save_dir = os.path.join(os.getcwd(), 'Models')
    save_dir = os.path.join(save_dir, 'adadepth')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, config.dataset)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if config.paper:
        load_dir = os.path.join(os.getcwd(), 'Models', 'Paper')
    else:
        load_dir = os.path.join(os.getcwd(), 'Models')

    if config.dataset == 'kitti':
        traindir_source = os.path.join(config.data_dir, 'vkitti')
        traindir_target = os.path.join(config.data_dir, 'kitti')
        validdir_target = os.path.join(config.data_dir, 'kitti')
        source_split = os.path.join(os.getcwd(), 'Splits', 'vkitti')
        source_split_file = os.path.join(source_split, 'vkitti_' + str(config.n_samples) + '_')
        target_split = os.path.join(os.getcwd(), 'Splits', config.dataset)
        target_split_file = os.path.join(target_split, config.dataset + '_' + str(config.n_samples) + '_')
    elif config.dataset == 'diml':
        traindir_source = os.path.join(config.data_dir, 'nyudepthv2', 'train')
        traindir_target = os.path.join(config.data_dir, 'dimlrgbd', 'dimlr_1500')
        validdir_target = os.path.join(config.data_dir, 'dimlrgbd', 'valid')
        source_split = os.path.join(os.getcwd(), 'Splits', 'nyu')
        source_split_file = os.path.join(source_split, 'nyu_' + str(config.n_samples) + '_')
        target_split = os.path.join(os.getcwd(), 'Splits', config.dataset)
        target_split_file = os.path.join(target_split, config.dataset + '_' + str(config.n_samples) + '_')
    else:
        raise NotImplementedError

    for j in range(args.n_rounds):
        model_dir = os.path.join(save_dir, config.arch + '_' + str(height) + 'x' + str(width) + '_'
                                 + str(config.n_samples) + '_' + str(j) + '.pt')
        print('saving model to: {}'.format(model_dir))

        if config.dataset == 'kitti':
            source_trainset = vKITTIDataset(traindir_source, src_file=source_split_file + str(j) + '.pickle',
                                            transform='valid', output_size=(height, width))

            target_trainset = KITTIDataset(traindir_target, src_file=target_split_file + str(j) + '.pickle',
                                           transform='valid', output_size=(height, width))

            target_validset = KITTIDataset(validdir_target, src_file=os.path.join(target_split, 'eigen_val.pickle'),
                                           transform='valid', output_size=(height, width))
        elif config.dataset == 'diml':
            source_trainset = NYUDataset(traindir_source, split='train',
                                         src_file=source_split_file + str(j) + '.pickle')
            target_trainset = DIMLRGBDDataset(traindir_target, split='train',
                                              src_file=target_split_file + str(j) + '.pickle')
            target_validset = DIMLRGBDDataset(validdir_target, split='val')
        else:
            raise NotImplementedError

        if config.arch == 'fastdepth':
            source_net = MobileNetSkipAdd(pretrained=False)
            target_net = MobileNetSkipAdd(pretrained=False)
        elif config.arch == 'resnet-upproj':
            source_net = ResNet(layers=50, decoder="upproj", output_size=(height, width))
            target_net = ResNet(layers=50, decoder="upproj", output_size=(height, width))
        else:
            raise NotImplementedError

        if config.dataset == 'kitti':
            checkpoint = torch.load(os.path.join(load_dir,
                                                 'vkitti', config.arch + '_' + str(height) + 'x' + str(width) + '.pt'))
        elif config.dataset == 'diml':
            checkpoint = torch.load(os.path.join(load_dir,
                                                 'nyu', config.arch + '_' + str(height) + 'x' + str(width) + '.pt'))
        else:
            raise NotImplementedError

        source_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(checkpoint['model_state_dict'])

        source_net.to(config.device)
        target_net.to(config.device)

        # fix source parameters
        set_requires_grad(source_net, False)

        # fix decoder parameters
        set_requires_grad(target_net, False)

        delta1, delta2, delta3, rmse, _ = test_depth_model(source_net, target_validset, config.device,
                                                           config.MIN_DEPTH, config.MAX_DEPTH,
                                                           output_sf=config.output_sf)
        print('Delta1: ', delta1)
        best_delta = delta1
        best_delta_rmse = rmse

        # create latent space discriminator
        if config.dataset == 'kitti':
            disc_z = Discriminator_Latent_Space_KITTI(resolution=config.resolution)
        elif config.dataset == 'diml':
            disc_z = Discriminator_Latent_Space_DIML()
        else:
            raise NotImplementedError
        disc_z.apply(weights_init_normal)
        disc_z.to(config.device)

        # create depth map discriminator
        disc_depth = PatchGAN_Discriminator()
        disc_depth.apply(weights_init_normal)
        disc_depth.to(config.device)

        if config.dataset == 'kitti':
            enc_optimizer = torch.optim.SGD(target_net.parameters(), lr=config.enc_lr,
                                            momentum=config.momentum)
            disc_z_optimizer = torch.optim.SGD(disc_z.parameters(), lr=config.disc_z_lr,
                                               momentum=config.momentum)
            disc_depth_optimizer = torch.optim.SGD(disc_depth.parameters(), lr=config.disc_d_lr,
                                                   momentum=config.momentum)
        elif config.dataset == 'diml':
            enc_optimizer = torch.optim.Adam(target_net.parameters(), lr=config.enc_lr,
                                             betas=(config.beta1, config.beta2))
            disc_z_optimizer = torch.optim.Adam(disc_z.parameters(), lr=config.disc_z_lr,
                                                betas=(config.beta1, config.beta2))
            disc_depth_optimizer = torch.optim.Adam(disc_depth.parameters(), lr=config.disc_d_lr,
                                                    betas=(config.beta1, config.beta2))
        else:
            raise NotImplementedError

        adadepth_training(config.n_epochs, source_trainset, target_trainset, target_validset, source_net, target_net,
                          disc_z, disc_depth, disc_z_optimizer, enc_optimizer, disc_depth_optimizer, patch_height,
                          patch_width, config.reg_val, config.device, model_dir, config.MIN_DEPTH, config.MAX_DEPTH,
                          config.batch_size, config.divisor, sf=config.output_sf, arch=config.arch)
