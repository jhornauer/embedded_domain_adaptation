import numpy as np
import torch
import torch.nn as nn
import time
import math
from torch.utils.data import DataLoader


#######################
def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v
    return res_dict


def setattr_cls_from_kwargs(cls, kwargs):
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def dict2clsattr(config):
    cfgs = {}
    for k, v in config.items():
        cfgs[k] = v

    class cfg_container: pass

    cfg_container.config = config
    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container


def merge_dicts(dict_a, dict_b):
    dict_a = flatten_dict(dict_a)
    dict_a.update({k: v for k, v in dict_b.items() if v is not None})
    return dict_a
##########################


def weights_init_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def set_requires_grad_encoder_true_fastdepth(model):
    for name, param in model.named_parameters():
        for i in range(10, 14):
            if ('conv' + str(i)) in name:
                param.requires_grad = True


def set_requires_grad_encoder_true_resnet(model):
    for name, param in model.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
        if 'layer' not in name and 'conv2' in name:
            param.requires_grad = True


def adadepth_training(n_epochs, src_trainset, trg_trainset, trg_validset, src_net, trg_net, disc_z, disc_d,
                      disc_z_optim, enc_optim, disc_d_optim, patch_height, patch_width, reg_val, device, model_path,
                      MIN_DEPTH, MAX_DEPTH, batch_size, divisor, sf=1.0, arch='fastdepth'):
    disc_d_criterion = nn.MSELoss()
    disc_z_criterion = nn.MSELoss()
    reg_criterion = nn.L1Loss()

    if arch == 'fastdepth':
        set_requires_grad_encoder_true = set_requires_grad_encoder_true_fastdepth
    elif arch == 'resnet-upproj':
        set_requires_grad_encoder_true = set_requires_grad_encoder_true_resnet
    else:
        raise NotImplementedError

    disc_z_losses = []
    disc_d_losses = []
    enc_losses = []
    delta1_vals = []
    rmse_vals = []

    r_loss_disc_d = 0
    r_loss_disc_z = 0
    r_loss_enc = 0

    best_delta = 0

    batch_size = int(batch_size / divisor)

    src_loader = DataLoader(src_trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    trg_loader = DataLoader(trg_trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    for e in range(n_epochs):

        disc_d_optim.zero_grad()
        disc_z_optim.zero_grad()
        enc_optim.zero_grad()

        for i, ((src_images, src_labels), (trg_images, trg_labels)) in enumerate(zip(src_loader, trg_loader)):

            # DISCRIMINATOR UPDATE STEP
            # fix generator weights
            set_requires_grad(trg_net, requires_grad=False)
            set_requires_grad(disc_z, requires_grad=True)
            set_requires_grad(disc_d, requires_grad=True)

            disc_z.train()
            disc_d.train()

            src_images = src_images.to(device)
            trg_images = trg_images.to(device)
            src_labels = src_labels.to(device)

            # rgb input
            input_d = torch.cat([src_images, trg_images])

            # depth map and latent space predictions
            src_d, src_z = src_net(src_images)
            trg_d, trg_z = trg_net(trg_images)

            # input for latent space discriminator and depth discriminator
            features_z = torch.cat([src_z, trg_z])
            # use source ground truth depth map instead of depth map prediction from source network
            features_d = torch.cat([src_labels, trg_d])

            # depth discriminator labels
            depth_labels_real = torch.ones((src_images.shape[0], 1, patch_height, patch_width), device=device)
            depth_labels_fake = torch.zeros((trg_images.shape[0], 1, patch_height, patch_width), device=device)

            depth_labels = torch.cat([depth_labels_real, depth_labels_fake])

            # latent space discriminator labels
            z_labels_real = torch.ones((src_images.shape[0], 1), device=device)
            z_labels_fake = torch.zeros((trg_images.shape[0], 1), device=device)

            z_labels = torch.cat([z_labels_real, z_labels_fake])

            # depth discriminator: which depth maps are real or fake ?
            y_d = disc_d(input_d, features_d)

            # latent space discriminator: which latent space is real or fake ?
            y_z = disc_z(features_z)

            loss_d = disc_d_criterion(y_d, depth_labels)
            loss_z = disc_z_criterion(y_z, z_labels)
            r_loss_disc_d += loss_d.item()
            r_loss_disc_z += loss_z.item()

            loss_d.backward()
            if (i+1) % divisor == 0:
                disc_d_optim.step()
                disc_d_optim.zero_grad()
            loss_z.backward()
            if (i + 1) % divisor == 0:
                disc_z_optim.step()
                disc_z_optim.zero_grad()

            # ENCODER UPDATE STEP
            # fix discriminator weights
            set_requires_grad(trg_net, False)
            set_requires_grad_encoder_true(trg_net)
            set_requires_grad(disc_d, requires_grad=False)
            set_requires_grad(disc_z, requires_grad=False)

            trg_net.train()

            trg_images = trg_images.to(device)

            # depth map and latent space predictions from source and target network for target domain
            trg_d_src, trg_z_src = src_net(trg_images)
            trg_d, trg_z = trg_net(trg_images)

            # flipped labels to fool discriminator
            d_labels_flipped = torch.ones((trg_images.shape[0], 1, patch_height, patch_width), device=device)
            z_labels_flipped = torch.ones((trg_images.shape[0], 1), device=device)

            # depth discriminator: classifiy target depth maps
            y_d = disc_d(trg_images, trg_d)

            # latent space discriminator: classify target latent space
            y_z = disc_z(trg_z)

            # calculate depth loss with flipped labels
            loss_d = disc_d_criterion(y_d, d_labels_flipped)
            # calculate latent space loss with flipped labels
            loss_z = disc_z_criterion(y_z, z_labels_flipped)

            # l1 regularization: distance between source network latent space and target network latent space
            loss_reg = reg_criterion(trg_z_src, trg_z)

            loss_enc = loss_d + loss_z + reg_val * loss_reg

            r_loss_enc += loss_reg.item()

            loss_enc.backward()
            if (i + 1) % divisor == 0:
                enc_optim.step()
                enc_optim.zero_grad()

        disc_d_losses.append(r_loss_disc_d / (len(src_loader) + len(trg_loader)))
        disc_z_losses.append(r_loss_disc_z / (len(src_loader) + len(trg_loader)))
        enc_losses.append(r_loss_enc / len(trg_loader))

        r_loss_disc_d = 0
        r_loss_disc_z = 0
        r_loss_enc = 0

        delta1, delta2, delta3, rmse, _ = test_depth_model(trg_net, trg_validset, device, MIN_DEPTH, MAX_DEPTH,
                                                           output_sf=sf)

        print('Epoch {} -- validation delta1: {:.3f}'.format(e, delta1))
        delta1_vals.append(delta1)
        rmse_vals.append(rmse)

        if delta1 > best_delta:
            best_delta = delta1
            torch.save({'epoch': e,
                        'model_state_dict': trg_net.state_dict()}, model_path)


def train_depth_model(model, criterion, optimizer, scheduler, n_epochs, train_dataset, device, model_dir, val_dataset,
                      MIN_DEPTH, MAX_DEPTH, batch_size, output_sf=1.0):
    model.to(device)
    loss_ges = 0
    delta1_ges = 0
    best_delta = 0
    for e in range(n_epochs):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.to(device, dtype=torch.float)
            target = target.to(device)
            output, _ = model(input)

            loss = criterion(output, target)
            loss_ges += loss.item()
            valid_mask = ((target > 0) + (output > 0)) > 0
            output = output[valid_mask]
            target = target[valid_mask]
            maxRatio = torch.max(output / target, target / output)
            delta1 = float((maxRatio < 1.25).float().mean())
            delta1_ges += delta1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_ges = loss_ges / len(train_loader)
        delta1_ges = delta1_ges / len(train_loader)
        print('Epoch {}: Loss: {:.3f} -- Delta1: {:.3f}'.format(e, loss_ges, delta1_ges))

        loss_ges = 0
        delta1_ges = 0

        delta1, delta2, delta3, rmse, _ = test_depth_model(model, val_dataset, device, MIN_DEPTH, MAX_DEPTH,
                                                           output_sf=output_sf)
        if delta1 > best_delta:
            best_delta = delta1
            print('Saved checkpoint with validation delta1: {:.3f}'.format(delta1))
            torch.save({'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'loss': loss_ges}, model_dir)


def test_depth_model(model, dataset, device, MIN_DEPTH, MAX_DEPTH, scaled=False, output_sf=1.0):
    rmse_ges = 0
    delta1_ges = 0
    delta2_ges = 0
    delta3_ges = 0

    model.eval()

    time_total = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

    for i, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)

        with torch.no_grad():
            start = time.time()
            output, _ = model(input)
            stop = time.time()
            time_total += (stop - start)
            output = output * output_sf

        rmse, delta1, delta2, delta3 = calc_metrics(output, target, MIN_DEPTH, MAX_DEPTH, median_scaled=scaled)

        rmse_ges += rmse
        delta1_ges += delta1
        delta2_ges += delta2
        delta3_ges += delta3

    avg_time = time_total / len(dataloader.dataset)
    avg_rmse = rmse_ges / len(dataloader.dataset)
    avg_delta1 = delta1_ges / len(dataloader.dataset)
    avg_delta2 = delta2_ges / len(dataloader.dataset)
    avg_delta3 = delta3_ges / len(dataloader.dataset)

    return avg_delta1, avg_delta2, avg_delta3, avg_rmse, avg_time


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


def calc_metrics(output, label, MIN_DEPTH, MAX_DEPTH, median_scaled=False):
    valid_mask = ((label > 0) + (output > 0)) > 0
    output = output[valid_mask]
    label = label[valid_mask]

    valid_mask_min = label > MIN_DEPTH
    label = label[valid_mask_min]
    output = output[valid_mask_min]

    valid_mask_max = label < MAX_DEPTH
    label = label[valid_mask_max]
    output = output[valid_mask_max]

    if median_scaled:
        scale = torch.median(label) / torch.median(output)
        output = output * scale

    output[output < MIN_DEPTH] = MIN_DEPTH
    output[output > MAX_DEPTH] = MAX_DEPTH

    mse = float(((label - output) ** 2).mean())
    rmse = math.sqrt(mse)

    maxRatio = torch.max(output / label, label / output)
    delta1 = float((maxRatio < 1.25).float().mean())
    delta2 = float((maxRatio < 1.25 ** 2).float().mean())
    delta3 = float((maxRatio < 1.25 ** 3).float().mean())

    return rmse, delta1, delta2, delta3
