import os
import json
import argparse
import torch
from utils import test_depth_model
from utils import merge_dicts, dict2clsattr
from data.kitti import KITTIDataset
from data.dimlrgbd import DIMLRGBDDataset
from networks.fastdepth import MobileNetSkipAdd, ResNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='./configs/kitti_fastdepth_low_1000.json')
    parser.add_argument('-device', type=str, default='cuda:0')
    parser.add_argument('-n_rounds', type=int, default=5)
    parser.add_argument('-scaled', action='store_true')
    parser.add_argument('-paper', action='store_true', help='load results from the paper')
    parser.add_argument('-data_dir', type=str)
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            model_config = json.load(f)
        train_config = vars(args)
    else:
        raise NotImplementedError

    config_dict = merge_dicts(model_config, train_config)
    config = dict2clsattr(config_dict)

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

    if config.dataset == 'kitti':
        test_dir = os.path.join(config.data_dir, 'kitti')
        test_split = os.path.join(os.getcwd(), 'Splits', config.dataset)
        testset = KITTIDataset(test_dir, src_file=os.path.join(test_split, 'eigen_test.pickle'), transform='valid',
                               output_size=(height, width))
    elif config.dataset == 'diml':
        test_dir = os.path.join(config.data_dir, 'dimlrgbd', 'test', 'HR')
        testset = DIMLRGBDDataset(test_dir, split='val')
    else:
        raise NotImplementedError

    root_dir = os.getcwd() + '/Models/'
    if config.paper:
        root_dir = root_dir + 'Paper/'

    if config.dataset == 'kitti':
        source_dir = root_dir + 'vkitti/' + config.arch + '_' + str(height) + 'x' + str(width) + '.pt'
        target_dir = root_dir + 'adadepth/kitti/' + config.arch + '_' + str(height) + 'x' + str(width) + '_' \
                     + str(config.n_samples)
    elif config.dataset == 'diml':
        source_dir = root_dir + 'nyu/' + config.arch + '_' + str(height) + 'x' + str(width) + '.pt'
        target_dir = root_dir + 'adadepth/diml/' + config.arch + '_' + str(height) + 'x' + str(width) + '_' \
                     + str(config.n_samples)
    else:
        raise NotImplementedError

    device = torch.device(config.device)

    delta1_total, delta2_total, delta3_total, rmse_total = 0, 0, 0, 0

    if config.arch == 'fastdepth':
        source_model = MobileNetSkipAdd()
        target_model = MobileNetSkipAdd()
    elif config.arch == 'resnet-upproj':
        source_model = ResNet(layers=50, decoder="upproj", output_size=(height, width))
        target_model = ResNet(layers=50, decoder="upproj", output_size=(height, width))
    else:
        raise NotImplementedError

    # Test source model
    source_checkpoint = torch.load(source_dir)
    source_model.load_state_dict(source_checkpoint['model_state_dict'])
    source_model.to(device)

    delta1, delta2, delta3, rmse, _ = test_depth_model(source_model, testset, device, config.MIN_DEPTH,
                                                       config.MAX_DEPTH, output_sf=config.output_sf)

    print('##################')
    print('## SOURCE MODEL ##')
    print('Delta1: {:.3f}'.format(delta1))
    print('Delta2: {:.3f}'.format(delta2))
    print('Delta3: {:.3f}'.format(delta3))
    print('RMSE: {:.3f}'.format(rmse))
    print('##################')

    for i in range(config.n_rounds):
        target_checkpoint = torch.load(target_dir + '_' + str(i) + '.pt')
        target_model.load_state_dict(target_checkpoint['model_state_dict'])
        target_model.to(device)

        delta1, delta2, delta3, rmse, _ = test_depth_model(target_model, testset, device, config.MIN_DEPTH,
                                                           config.MAX_DEPTH, config.scaled, output_sf=config.output_sf)

        delta1_total += delta1
        delta2_total += delta2
        delta3_total += delta3
        rmse_total += rmse

    print('\n')
    print('##################')
    print('## TARGET MODEL ##')
    print('Delta1: {:.3f}'.format(delta1_total / config.n_rounds))
    print('Delta2: {:.3f}'.format(delta2_total / config.n_rounds))
    print('Delta3: {:.3f}'.format(delta3_total / config.n_rounds))
    print('RMSE: {:.3f}'.format(rmse_total / config.n_rounds))
    print('##################')
