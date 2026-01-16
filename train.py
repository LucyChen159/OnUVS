from tqdm import trange
import torch
import numpy as np
import os
import math
from PIL import Image

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from logger import Logger
from modules.model import DiscriminatorFullModel,  GeneratorFullModel_kp_texture

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater

from tqdm import tqdm

def train(config, generator, discriminator,bg_predictor, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    optimizer_bg_predictor = None
    if bg_predictor:
        optimizer_bg_predictor = torch.optim.Adam(
            [{'params':bg_predictor.parameters(),'initial_lr': train_params['lr_generator']}], 
            lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, bg_predictor, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,optimizer_bg_predictor, 
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if bg_predictor:
        scheduler_bg_predictor = MultiStepLR(optimizer_bg_predictor, train_params['epoch_milestones'],
                                              gamma=0.1, last_epoch=start_epoch - 1)


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel_kp_texture(kp_detector, generator, bg_predictor, discriminator, train_params, device_ids)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params, device_ids)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    predictio_asve_path = "/data3/rs/project/video/heart_vit_bg/prediction_bg"
    if not os.path.exists(predictio_asve_path):
        os.makedirs(predictio_asve_path)

    bg_start = train_params['bg_start']
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            n_o = 0
            for x in tqdm(dataloader):  # driving (batchsize, 1, 256, 256), source (batchsize, 1, 256, 256)
                losses_generator, generated = generator_full(x, epoch)  
                #可视化occlusion map在训练过程中的变化
                if 'prediction' in generated.keys() and n_o % 500 == 0:
                    source = x['source'][0].data.cpu().repeat(1, 3, 1, 1)
                    source = np.transpose(source, [0, 2, 3, 1])
                    prediction = generated['prediction'].data.cpu().repeat(1, 3, 1, 1)
                    prediction = torch.nn.functional.interpolate(prediction, size=source.shape[1:3]).numpy()
                    prediction = np.transpose(prediction, [0, 2, 3, 1])

                    image_occlusion = prediction[0]
                    image_occlusion = (image_occlusion/np.max(image_occlusion)) * 250
                    occlusin_name = 'epoch_' + str(epoch) + "_prediction_" + str(n_o)+'.png'
                    image_save_path = os.path.join(predictio_asve_path, occlusin_name)
                    image_occlusion = Image.fromarray(image_occlusion[:,:,0])
                    if image_occlusion.mode == "F":
                        image_occlusion = image_occlusion.convert('RGB')
                    n_o = n_o + 1
                    image_occlusion.save(image_save_path)
                else:
                    n_o = n_o + 1

                # if n_o % 300 == 0:
                #     print('epoch_' + str(epoch) + '_' + "prediction_" + str(n_o))

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()

                clip_grad_norm_(kp_detector.parameters(), max_norm=10, norm_type = math.inf)
                clip_grad_norm_(generator.parameters(), max_norm=10, norm_type = math.inf)

                if bg_predictor and epoch>=bg_start:
                    clip_grad_norm_(bg_predictor.parameters(), max_norm=10, norm_type = math.inf)

                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                if bg_predictor and epoch>=bg_start:
                    optimizer_bg_predictor.step()
                    optimizer_bg_predictor.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                 
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            if bg_predictor:
                scheduler_bg_predictor.step() 
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'bg_predictor': bg_predictor,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_bg_detector': optimizer_kp_detector}, inp=x, out=generated)
