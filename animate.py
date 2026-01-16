import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import FramesDatasetWithJson_single
from logger import  Logger, Logger_cal_16

from scipy.spatial import ConvexHull
import numpy as np

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def animate(config, generator, kp_detector, bg_predictor, checkpoint, log_dir):
    animate_params = config['animate_params']

    json_dir = 'heart_data_final/test'
    result_root = "heart_data_final/test/epoch50_animate"

    dataset = FramesDatasetWithJson_single(root_dir=json_dir)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector,bg_predictor=bg_predictor)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    generator.eval()
    kp_detector.eval()
    bg_predictor.eval()

    for x in tqdm(dataloader):
        with torch.no_grad():
            predictions = []

            if torch.cuda.is_available():
                x['driving_video'] = x['video'].type(torch.FloatTensor).cuda()
                x['source'] = x['video'][:,:,0].type(torch.FloatTensor).cuda()
            driving_video = x['driving_video']
            source_frame = x['source']
            dring_name = x['d_name'][0].split('_gt')[0]

            kp_source = kp_detector(source_frame)
            kp_driving_initial = kp_detector(driving_video[:, :, 0])

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                bg_param = bg_predictor(source_frame, driving_frame)

                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm, bg_param = bg_param)

                out['kp_driving'] = kp_driving
                out['kp_source'] = kp_source
                out['kp_norm'] = kp_norm

                del out['sparse_deformed']

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                
                logger = Logger_cal_16(log_dir=result_root, name=f"{dring_name}", visualizer_params=config['visualizer_params'])

                logger.visualize_source_driving(driving_frame, source_frame, out, frame_idx)
                