#
# This is an implementation of the method described in paper
#
# Matthew Lee, Ozan Oktay, Andreas Schuh, Michiel Schaap, Ben Glocker
# Image-and-Spatial Transformer Networks for Structure-guided Image Registration
# In MICCAI 2019
#
# All rights reserved. Copyright 2019
#

import os
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

import matplotlib as mpl

back_end = mpl.get_backend()
try:
    mpl.use('module://backend_interagg')
    import matplotlib.pyplot as plt

    print('Set matplotlib backend to interagg')
except ImportError:
    print('Cannot set matplotlib backend to interagg, resorting to default backend {}'.format(back_end))
    mpl.use(back_end)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print('Cannot set matplotlib backend to interagg, resorting to default backend {}'.format(back_end))
    mpl.use(back_end)
    import matplotlib.pyplot as plt

import SimpleITK as sitk

from pymira.nets.itn import ITN2D, ITN3D
from pymira.nets.stn import STN2D, BSplineSTN2D, STN3D, BSplineSTN3D
from pymira.img.processing import zero_mean_unit_var
from pymira.img.processing import range_matching
from pymira.img.processing import zero_one
from pymira.img.processing import threshold_zero
from pymira.img.transforms import Resampler
from pymira.img.transforms import Normalizer
from pymira.img.datasets import ImageSegRegDataset
import pymira.utils.metrics as mira_metrics
import pymira.utils.tensorboard_helpers as mira_th
from tensorboardX import SummaryWriter
from attrdict import AttrDict

separator = '----------------------------------------'


def write_images(writer, phase, image_dict, n_iter, mode3d):
    for name, image in image_dict.items():
        if mode3d:
            writer.add_image('{}/{}'.format(phase, name), mira_th.volume_to_batch_image(image), n_iter)
        else:
            writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, :, :, :]), n_iter)


def write_values(writer, phase, value_dict, n_iter):
    for name, value in value_dict.items():
        writer.add_scalar('{}/{}'.format(phase, name), value, n_iter)


def set_up_model_and_preprocessing(phase, args):
    print(separator)
    print('Starting {}...'.format(phase))
    print(separator)

    with open(args.config) as f:
        config = json.load(f)

    print('Config from file: ' + str(config))

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")

    print('Device: ' + str(device))
    if use_cuda:
        print('GPU: ' + str(torch.cuda.get_device_name(int(args.dev))))

    if args.transformation == 'affine':
        if args.mode3d:
            stn_model = STN3D
        else:
            stn_model = STN2D
    elif args.transformation == 'bspline':
        if args.mode3d:
            stn_model = BSplineSTN3D
        else:
            stn_model = BSplineSTN2D
    else:
        raise NotImplementedError('transformation {} not supported'.format(args.transformation))

    resampler_img = Resampler(config['spacing'], config['size'])
    resampler_seg = Resampler(config['spacing'], config['size'])

    if config['normalizer_img'] == 'zero_mean_unit_var':
        normalizer_img = Normalizer(zero_mean_unit_var)
    elif config['normalizer_img'] == 'range_matching':
        normalizer_img = Normalizer(range_matching)
    elif config['normalizer_img'] == 'zero_one':
        normalizer_img = Normalizer(zero_one)
    elif config['normalizer_img'] == 'threshold_zero':
        normalizer_img = Normalizer(threshold_zero)
    elif config['normalizer_img'] == 'none':
        normalizer_img = None
    else:
        raise NotImplementedError('Normalizer {} not supported'.format(config['normalizer_img']))

    if config['normalizer_seg'] == 'zero_mean_unit_var':
        normalizer_seg = Normalizer(zero_mean_unit_var)
    elif config['normalizer_seg'] == 'range_matching':
        normalizer_seg = Normalizer(range_matching)
    elif config['normalizer_seg'] == 'zero_one':
        normalizer_seg = Normalizer(zero_one)
    elif config['normalizer_seg'] == 'threshold_zero':
        normalizer_seg = Normalizer(threshold_zero)
    elif config['normalizer_seg'] == 'none':
        normalizer_seg = None
    else:
        raise NotImplementedError('Normalizer {} not supported'.format(config['normalizer_seg']))

    if args.loss == 'e':
        loss = 'explicit'
    elif args.loss == 'i':
        loss = 'implicit'
    elif args.loss == 's':
        loss = 'supervised'
    elif args.loss == 'u':
        loss = 'unsupervised'
    else:
        raise NotImplementedError('Loss {} not supported'.format(args.loss))

    if args.mode3d:
        itn = ITN3D(input_channels=1).to(device)
    else:
        itn = ITN2D(input_channels=1).to(device)
    stn = stn_model(input_size=config['size'], input_channels=2, device=device).to(device)
    parameters = list(itn.parameters()) + list(stn.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])

    config_dict = {'config': config,
                   'device': device,
                   'normalizer_img': normalizer_img,
                   'normalizer_seg': normalizer_seg,
                   'resampler_img': resampler_img,
                   'resampler_seg': resampler_seg,
                   'stn': stn,
                   'itn': itn,
                   'optimizer': optimizer,
                   'loss': loss,
                   }
    print('File config: {}'.format(config_dict))

    return AttrDict(config_dict)


def process_batch(config, itn, stn, batch_samples):
    source, target = batch_samples['source'].to(config.device), batch_samples['target'].to(config.device)
    source_seg, target_seg = batch_samples['source_seg'].to(config.device), batch_samples['target_seg'].to(
        config.device)

    if itn is not None:
        source_prime = itn(source)
        target_prime = itn(target)
        if config.loss == 'unsupervised' or config.loss == 'supervised':
            source_prime = source
            target_prime = target
    else:
        source_prime = source
        target_prime = target

    stn(torch.cat((source_prime, target_prime), dim=1))
    warped_source = stn.warp_image(source)
    warped_source_prime = stn.warp_image(source_prime)
    warped_source_seg = stn.warp_image(source_seg)

    # Custom Metrics - thresholding at 0.5 is a bit arbitrarily and only makes sense if structure map is in [0,1]
    target_seg_binary = target_seg > 0.5
    warped_source_seg_binary = warped_source_seg > 0.5

    dice = mira_metrics.dice_score(warped_source_seg_binary, target_seg_binary, unindexed_classes=1)['1']
    hausdorff_distance = \
        mira_metrics.hausdorff_distance(warped_source_seg_binary, target_seg_binary, unindexed_classes=1, spacing=config.config.spacing)[
            '1']
    average_surface_distance = \
        mira_metrics.average_surface_distance(warped_source_seg_binary, target_seg_binary, unindexed_classes=1, spacing=config.config.spacing)['1']
    precision = mira_metrics.precision(warped_source_seg_binary, target_seg_binary, unindexed_classes=1)['1']
    recall = mira_metrics.recall(warped_source_seg_binary, target_seg_binary, unindexed_classes=1)['1']

    # General Loss Calculation
    loss_itn = F.mse_loss(source_prime, source_seg) + F.mse_loss(target_prime, target_seg)
    loss_stn_u = F.mse_loss(warped_source, target)
    loss_stn_s = F.mse_loss(warped_source_seg, target_seg)
    loss_stn_i = F.mse_loss(warped_source_prime, target_seg) + F.mse_loss(warped_source_seg, target_prime)
    loss_stn_r = F.mse_loss(warped_source_prime, target_prime)

    if config.loss == 'explicit':
        loss_train = loss_itn + loss_stn_s      # ISTN-e
    elif config.loss == 'implicit':
        loss_train = loss_stn_i + loss_stn_s    # ISTN-i
    elif config.loss == 'supervised':
        loss_train = loss_stn_s                 # STN-s
    elif config.loss == 'unsupervised':
        loss_train = loss_stn_u                 # STN-u
    else:
        raise NotImplementedError('Loss {} not supported'.format(config.loss))

    values_dict = {'loss_itn': loss_itn,
                   'loss_stn_u': loss_stn_u,
                   'loss_stn_s': loss_stn_s,
                   'loss_stn_i': loss_stn_i,
                   'loss_stn_r': loss_stn_r,
                   'loss': loss_train,
                   'metric_dice': dice,
                   'metric_hd': hausdorff_distance,
                   'metric_asd': average_surface_distance,
                   'metric_precision': precision,
                   'metric_recall': recall}

    images_dict = {'source': source,
                  'source_prime': source_prime,
                  'source_seg': source_seg,
                  'target': target,
                  'target_prime': target_prime,
                  'target_seg': target_seg,
                  'warped_source': warped_source,
                  'warped_source_prime': warped_source_prime,
                  'warped_source_seg': warped_source_seg}

    return loss_train, images_dict, values_dict


def train(args):
    config = set_up_model_and_preprocessing('TRAINING', args)

    writer = SummaryWriter('{}/tensorboard'.format(args.out))
    global_step = 0

    print(separator)
    print('TRAINING data...')
    print(separator)

    dataset_train = ImageSegRegDataset(args.train, args.train_seg, args.train_msk, normalizer_img=config.normalizer_img,
                                       normalizer_seg=config.normalizer_seg, resampler_img=config.resampler_img,
                                       resampler_seg=config.resampler_seg)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.config['batch_size'], shuffle=True)

    if args.val is not None:
        print(separator)
        print('VALIDATION data...')
        print(separator)
        dataset_val = ImageSegRegDataset(args.val, args.val_seg, args.val_msk, normalizer_img=config.normalizer_img,
                                         normalizer_seg=config.normalizer_seg, resampler_img=config.resampler_img,
                                         resampler_seg=config.resampler_seg)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    # Create output directory
    out_dir = os.path.join(args.out, 'train')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.save_temp:
        temp_dir = os.path.join(out_dir, 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        for idx in range(0, len(dataset_train)):
            sample = dataset_train.get_sample(idx)
            sitk.WriteImage(sample['source'], os.path.join(temp_dir, 'sample_' + str(idx) + '_source.nii.gz'))
            sitk.WriteImage(sample['target'], os.path.join(temp_dir, 'sample_' + str(idx) + '_target.nii.gz'))
            sitk.WriteImage(sample['source_seg'], os.path.join(temp_dir, 'sample_' + str(idx) + '_source_seg.nii.gz'))
            sitk.WriteImage(sample['target_seg'], os.path.join(temp_dir, 'sample_' + str(idx) + '_target_seg.nii.gz'))

    print(separator)

    # Note: Must match those used in process_batch()
    loss_names = ['loss_itn', 'loss_stn_u', 'loss_stn_s', 'loss_stn_i', 'loss_stn_r', 'loss', 'metric_dice',
                  'metric_hd', 'metric_asd', 'metric_precision', 'metric_recall']
    train_logger = mira_metrics.Logger('TRAIN', loss_names)
    validation_logger = mira_metrics.Logger('VALID', loss_names)

    model_dir = os.path.join(out_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in range(1, config.config['epochs'] + 1):
        config.stn.train()
        config.itn.train()

        # Training
        for batch_idx, batch_samples in enumerate(tqdm(dataloader_train, desc='Epoch {}'.format(epoch))):
            global_step += 1
            config.optimizer.zero_grad()
            loss, images_dict, values_dict = process_batch(config, config.itn, config.stn, batch_samples)
            loss.backward()
            config.optimizer.step()
            train_logger.update_epoch_logger(values_dict)

        train_logger.update_epoch_summary(epoch)
        write_values(writer, 'train', value_dict=train_logger.get_latest_dict(), n_iter=global_step)
        write_images(writer, 'train', image_dict=images_dict, n_iter=global_step, mode3d=args.mode3d)

        # Validation
        if args.val is not None and (epoch == 1 or epoch % config.config['val_interval'] == 0):
            config.stn.eval()
            config.itn.eval()

            with torch.no_grad():
                for batch_idx, batch_samples in enumerate(dataloader_val):
                    loss, images_dict, values_dict = process_batch(config, config.itn, config.stn, batch_samples)
                    validation_logger.update_epoch_logger(values_dict)

            validation_logger.update_epoch_summary(epoch)
            write_values(writer, phase='val', value_dict=validation_logger.get_latest_dict(), n_iter=global_step)
            write_images(writer, phase='val', image_dict=images_dict, n_iter=global_step, mode3d=args.mode3d)

            print(separator)
            train_logger.print_latest()
            validation_logger.print_latest()
            print(separator)

            torch.save(config.itn.state_dict(), model_dir + '/itn_' + str(epoch) + '.pt')
            torch.save(config.stn.state_dict(), model_dir + '/stn_' + str(epoch) + '.pt')

    torch.save(config.itn.state_dict(), model_dir + '/itn.pt')
    torch.save(config.stn.state_dict(), model_dir + '/stn.pt')

    print(separator)
    print('Finished TRAINING... Plotting Graphs\n\n')
    for loss_name, colour in zip(['loss'], ['b']):
        plt.plot(train_logger.epoch_number_logger, train_logger.epoch_summary[loss_name], c=colour,
                 label='train {}'.format(loss_name))
        plt.plot(validation_logger.epoch_number_logger, validation_logger.epoch_summary[loss_name], c=colour,
                 linestyle=':',
                 label='val {}'.format(loss_name))

    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def test(args):
    config = set_up_model_and_preprocessing('TESTING', args)

    dataset_test = ImageSegRegDataset(args.test, args.test_seg, args.test_msk, normalizer_img=config.normalizer_img,
                                      normalizer_seg=config.normalizer_seg, resampler_img=config.resampler_img,
                                      resampler_seg=config.resampler_seg)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)
    loss_names = ['loss_itn', 'loss_stn_u', 'loss_stn_s', 'loss_stn_i', 'loss_stn_r', 'loss', 'metric_dice',
                  'metric_hd', 'metric_asd', 'metric_precision', 'metric_recall']
    test_logger = mira_metrics.Logger('TEST', loss_names)

    # Create output directory
    out_dir = os.path.join(args.out, 'test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    config.itn.load_state_dict(torch.load(args.model + '/itn.pt'))
    config.itn.eval()

    config.stn.load_state_dict(torch.load(args.model + '/stn.pt'))
    config.stn.eval()

    with torch.no_grad():
        for index, batch_samples in enumerate(dataloader_test):
            loss, images_dict, values_dict = process_batch(config, config.itn, config.stn, batch_samples)
            test_logger.update_epoch_logger(values_dict)

            source_transformed = sitk.GetImageFromArray(images_dict['source_prime'].cpu().squeeze().numpy())
            source_transformed.CopyInformation(dataset_test.get_sample(index)['source'])
            sitk.WriteImage(source_transformed,
                            os.path.join(out_dir, 'sample_' + str(index) + '_source_prime.nii.gz'))

            target_transformed = sitk.GetImageFromArray(images_dict['target_prime'].cpu().squeeze().numpy())
            target_transformed.CopyInformation(dataset_test.get_sample(index)['target'])
            sitk.WriteImage(target_transformed,
                            os.path.join(out_dir, 'sample_' + str(index) + '_target_prime.nii.gz'))

            warped_source = sitk.GetImageFromArray(images_dict['warped_source'].cpu().squeeze().numpy())
            warped_source.CopyInformation(dataset_test.get_sample(index)['target'])
            sitk.WriteImage(warped_source,
                            os.path.join(out_dir, 'sample_' + str(index) + '_warped_source.nii.gz'))

            warped_source_seg = sitk.GetImageFromArray(images_dict['warped_source_seg'].cpu().squeeze().numpy())
            warped_source_seg.CopyInformation(dataset_test.get_sample(index)['target'])
            sitk.WriteImage(warped_source_seg,
                            os.path.join(out_dir, 'sample_' + str(index) + '_warped_source_seg.nii.gz'))

            sitk.WriteImage(dataset_test.get_sample(index)['source'],
                            os.path.join(out_dir, 'sample_' + str(index) + '_source.nii.gz'))
            sitk.WriteImage(dataset_test.get_sample(index)['target'],
                            os.path.join(out_dir, 'sample_' + str(index) + '_target.nii.gz'))
            sitk.WriteImage(dataset_test.get_sample(index)['source_seg'],
                            os.path.join(out_dir, 'sample_' + str(index) + '_source_seg.nii.gz'))
            sitk.WriteImage(dataset_test.get_sample(index)['target_seg'],
                            os.path.join(out_dir, 'sample_' + str(index) + '_target_seg.nii.gz'))
        with open(os.path.join(out_dir,'test_results.yml'), 'w') as outfile:
            yaml.dump(test_logger.get_epoch_logger(), outfile)
    test_logger.update_epoch_summary(0)

    if args.no_refine == False:
        refine_config = set_up_model_and_preprocessing('REFINEMENT', args)
        config.itn.eval()

        for index, batch_samples in enumerate(dataloader_test):

            print('Processing image ' + str(index+1) + ' of ' + str(len(dataset_test)))

            # Set up fine tuning network to have grads but not the stn
            refine_config.stn.load_state_dict(torch.load(args.model + '/stn.pt'))
            refine_config.stn.train()

            optimizer = torch.optim.Adam(refine_config.stn.parameters(), lr=refine_config.config['learning_rate'])

            # Fine tune STN
            for epoch in range(1, config.config['refine'] + 1):
                optimizer.zero_grad()
                _loss, images_dict, values_dict = process_batch(config, config.itn, refine_config.stn, batch_samples)
                loss = values_dict['loss_stn_r']
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                loss, images_dict, values_dict = process_batch(config, config.itn, refine_config.stn, batch_samples)
                test_logger.update_epoch_logger(values_dict)

                warped_source = sitk.GetImageFromArray(images_dict['warped_source'].cpu().squeeze().numpy())
                warped_source.CopyInformation(dataset_test.get_sample(index)['target'])
                sitk.WriteImage(warped_source,
                                os.path.join(out_dir, 'sample_' + str(index) + '_warped_source_refined.nii.gz'))

                warped_source_seg = sitk.GetImageFromArray(images_dict['warped_source_seg'].cpu().squeeze().numpy())
                warped_source_seg.CopyInformation(dataset_test.get_sample(index)['target'])
                sitk.WriteImage(warped_source_seg,
                                os.path.join(out_dir, 'sample_' + str(index) + '_warped_source_seg_refined.nii.gz'))
            with open(os.path.join(out_dir, 'test_results_refined.yml'), 'w') as outfile:
                yaml.dump(test_logger.get_epoch_logger(), outfile)


if __name__ == '__main__':

    output_dir = 'output'
    model_dir = output_dir + '/train/model'

    # Set up argument parser
    parser = argparse.ArgumentParser(description='ISTN registration')
    parser.add_argument('--save_temp', default=False, action='store_true', help='save temporary files (default: True)')
    parser.add_argument('--dev', default='0', help='cuda device (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # Data args
    parser.add_argument('--train', default='data/synth2d/train.csv', help='training data csv file')
    parser.add_argument('--train_seg', default='data/synth2d/train.seg.csv', help='training data csv file')
    parser.add_argument('--train_msk', default=None, help='training data csv file')
    parser.add_argument('--val', default='data/synth2d/val.csv', help='validation data csv file')
    parser.add_argument('--val_seg', default='data/synth2d/val.seg.csv', help='validation data csv file')
    parser.add_argument('--val_msk', default=None, help='validation data csv file')
    parser.add_argument('--test', default='data/synth2d/val.csv', help='testing data csv file')
    parser.add_argument('--test_seg', default='data/synth2d/val.seg.csv', help='testing data csv file')
    parser.add_argument('--test_msk', default=None, help='testing data csv file')

    # Logging args
    parser.add_argument('--out', default=output_dir, help='output root directory')
    parser.add_argument('--model', default=model_dir, help='model directory')

    # Network args
    parser.add_argument('--mode3d', default=False, action='store_true', help='enable 3D mode', )
    parser.add_argument('--config', default="data/synth2d/config.json", help='config file')

    parser.add_argument('--loss', default="u",
                        help='loss type, u=unsupervised, s=supervised, e=explicit, i=implicit',
                        choices=['u', 's', 'e', 'i'])
    parser.add_argument('--transformation', type=str, default='affine', help='transformation model',
                        choices=['affine', 'bspline'])
    parser.add_argument('--no_refine', default=False, action='store_true', help='disable iterative refinement', )

    args = parser.parse_args()

    # Run training
    if args.train is not None:
        train(args)

    # Run testing
    if args.test is not None:
        test(args)

    # EXAMPLE USAGE FOR 2D SYNTHETIC DATA
    #
    # STN-u (unsupervised)
    # python istn-reg.py --config data/synth2d/config.json --transformation affine --loss u --out output/stn-u --model output/stn-u/train/model
    #
    # STN-s (supervised)
    # python istn-reg.py --config data/synth2d/config.json --transformation affine --loss s --out output/stn-s --model output/stn-s/train/model
    #
    # ISTN-e (explicit)
    # python istn-reg.py --config data/synth2d/config.json --transformation affine --loss e --out output/stn-e --model output/stn-e/train/model
    #
    # ISTN-i (implicit)
    # python istn-reg.py --config data/synth2d/config.json --transformation affine --loss i --out output/stn-i --model output/stn-i/train/model
    #
    #
    # EXAMPLE USAGE FOR 3D BRAIN REGISTRATION
    #
    # ISTN - i(implicit)
    # python istn-reg.py --mode3d --loss i --out output3d/istn-i --model output3d/istn-i/train/model --config data/brain3d/config.affine.json --train data/brain3d/train.csv --train_seg data/brain3d/train.seg.csv --train_msk data/brain3d/train.msk.csv --val data/brain3d/val.csv --val_seg data/brain3d/val.seg.csv --val_msk data/brain3d/val.msk.csv --test data/brain3d/test.csv --test_seg data/brain3d/test.seg.csv --test_msk data/brain3d/test.msk.csv
