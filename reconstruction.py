import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback


def reconstruction(config, generator, kp_detector, bg_predictor, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        bg_predictor = DataParallelWithCallback(bg_predictor)
        
    generator.eval()
    kp_detector.eval()
    bg_predictor.eval()

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            kp_source = kp_detector(x['video'][:, :, 0])
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                bg_param = bg_predictor(x['source'], x['driving'])
                out = generator(source, kp_source=kp_source, bg_param = bg_param, kp_driving=kp_driving)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize_rec(source=source,
                                                                                    driving=driving, out=out)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))

def reconstruction_every16(config, generator, kp_detector, checkpoint, log_dir, dataset):
    root_dir = 'reconstruction16_50epoch_mid'
    png_dir = os.path.join(log_dir, root_dir, 'png')
    gt_dir = os.path.join(log_dir, root_dir, 'gt_png')
    log_dir = os.path.join(log_dir, root_dir)


    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            GT = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            kp_source = kp_detector(x['video'][:, :, 0])
            source = x['video'][:, :, 0]
            for frame_idx in range(x['video'].shape[2]):
                if frame_idx % 16 == 0 and frame_idx != 0:
                    source = x['video'][:, :, frame_idx]
                    index = frame_idx // 16
                    predictions = np.concatenate(predictions, axis=1)
                    imageio.imsave(os.path.join(png_dir, x['name'][0].split('.')[0] + '_%03d.png'%index), (255 * predictions).astype(np.uint8))

                    image_name = x['name'][0].split('.')[0] + '_%03d'%index + config['reconstruction_params']['format']
                    imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

                    GT = np.concatenate(GT, axis=1)
                    imageio.imsave(os.path.join(gt_dir, x['name'][0].split('.')[0] + '_%03d.png'%index), (255 * GT).astype(np.uint8))

                    predictions = []
                    visualizations = []
                    GT = []                    

                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                GT.append(np.transpose(x['video'][:, :, frame_idx].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize_rec(source=source,
                                                                                    driving=driving, out=out)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

    print("Reconstruction loss: %s" % np.mean(loss_list))

def reconstruction_every16_new(config, generator, kp_detector, bg_predictor, checkpoint, log_dir, dataset):
    root_dir = 'reconstruction16_50epoch_mid'
    png_dir = os.path.join(log_dir, root_dir, 'png')
    gt_dir = os.path.join(log_dir, root_dir, 'gt_png')
    log_dir = os.path.join(log_dir, root_dir)


    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            GT = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            mid_index = x['video'].shape[2] // 2
            kp_source = kp_detector(x['video'][:, :, mid_index])
            source = x['video'][:, :, mid_index]
            for frame_idx in range(x['video'].shape[2]):
                if frame_idx % 16 == 0 and frame_idx != 0:
                    # source = x['video'][:, :, frame_idx+8]
                    index = frame_idx // 16
                    predictions = np.concatenate(predictions, axis=1)
                    imageio.imsave(os.path.join(png_dir, x['name'][0].split('.')[0] + '_%03d.png'%index), (255 * predictions).astype(np.uint8))

                    image_name = x['name'][0].split('.')[0] + '_%03d'%index + config['reconstruction_params']['format']
                    imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

                    GT = np.concatenate(GT, axis=1)
                    imageio.imsave(os.path.join(gt_dir, x['name'][0].split('.')[0] + '_%03d.png'%index), (255 * GT).astype(np.uint8))

                    predictions = []
                    visualizations = []
                    GT = []                    

                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                GT.append(np.transpose(x['video'][:, :, frame_idx].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize_rec(source=source,
                                                                                    driving=driving, out=out)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

    print("Reconstruction loss: %s" % np.mean(loss_list))