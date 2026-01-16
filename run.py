import matplotlib

matplotlib.use('Agg')

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import shutil

from frames_dataset import FramesDatasetWithJson_DDH

from modules.generator import OcclusionAwareGenerator_multi_texture
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
from modules.bg_motion_predictor import BGMotionPredictor

import torch

from train import train
from reconstruction import reconstruction
from animate import animate

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="OnUVS/config/A4C-256.yaml", help="path to config")
    parser.add_argument("--mode", default="animate", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='heart_vit_bg_result', help="path to log into")
    parser.add_argument("--checkpoint", default="00000099-checkpoint.pth.tar", help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    generator = OcclusionAwareGenerator_multi_texture(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector)
    
    bg_predictor = None
    if (config['model_params']['common_params']['bg']):
        bg_predictor = BGMotionPredictor()
        if torch.cuda.is_available():
            bg_predictor.to(opt.device_ids[0])


    dataset = FramesDatasetWithJson_DDH(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, bg_predictor, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, bg_predictor, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, bg_predictor = bg_predictor, checkpoint = opt.checkpoint, log_dir = log_dir)
