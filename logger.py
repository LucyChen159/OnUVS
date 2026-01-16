import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
# from skimage.draw import circle
from skimage.draw import disk
import cv2

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None, 
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None,bg_predictor=None):
        checkpoint = torch.load(checkpoint_path)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % 2 == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)

class Logger_demo:
    def __init__(self, log_dir, visualizer_params=None, zfill_num=8):

        self.loss_list = []
        self.log_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'save_kp')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.zfill_num = zfill_num
        self.visualizer = Visualizer_demo(**visualizer_params)

    def visualize_source_driving(self, driving, source, out, index, flag_sup=False):
        image = self.visualizer.visualize(driving, source, out, flag_sup=flag_sup)
        if flag_sup:
            path = os.path.join(self.visualizations_dir, 'sup')
            os.makedirs(path, exist_ok=True)
            imageio.imsave(os.path.join(path, "%s.png" % str(index).zfill(self.zfill_num)), image)
        else:
            path = os.path.join(self.visualizations_dir, 'unsup')
            os.makedirs(path, exist_ok=True)
            imageio.imsave(os.path.join(path, "%s.png" % str(index).zfill(self.zfill_num)), image)

    def visualize_video(self, save_path, flag_sup=False):
        os.makedirs(os.path.join(self.log_dir, save_path), exist_ok= True)
        if flag_sup:
            img_path_list = os.listdir(os.path.join(self.visualizations_dir, 'sup'))
            img_path_list.sort()
            img = cv2.imread(os.path.join(self.visualizations_dir, 'sup', img_path_list[0]))
            video_path = os.path.join(self.log_dir, save_path, "result_sup.mp4")
            img_root = os.path.join(self.visualizations_dir, 'sup')
        else:
            img_path_list = os.listdir(os.path.join(self.visualizations_dir, 'unsup'))
            img_path_list.sort()
            img = cv2.imread(os.path.join(self.visualizations_dir, 'unsup', img_path_list[0]))
            video_path = os.path.join(self.log_dir, save_path, "result_unsup.mp4")
            img_root = os.path.join(self.visualizations_dir, 'unsup')

        fps = 5.0  
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        videoWriter = cv2.VideoWriter(video_path, fourcc=fourcc, fps=fps, frameSize=(img.shape[1], img.shape[0]))

        for i in img_path_list:
            frame = cv2.imread(os.path.join(img_root, i))
            videoWriter.write(frame)            
        
        videoWriter.release()

class Logger_cal_16:
    def __init__(self, log_dir, name, visualizer_params=None, zfill_num=8):

        self.log_dir = log_dir
        self.pred_root = os.path.join(self.log_dir, 'prediction')
        self.gt_root = os.path.join(self.log_dir, 'GT')
        self.source_root = os.path.join(self.log_dir, 'source')
        os.makedirs(self.pred_root, exist_ok=True)
        os.makedirs(self.gt_root, exist_ok=True)
            
        self.name = name
        self.zfill_num = zfill_num
        self.visualizer = Visualizer_demo(**visualizer_params)

    def visualize_source_driving(self, driving, source, out, index):
        image, gt_image, source_image = self.visualizer.visualize_for_metrics(driving, source, out)
        pred_path = os.path.join(self.pred_root, self.name)
        os.makedirs(pred_path, exist_ok=True)
        imageio.imsave(os.path.join(pred_path, "%s.png" % str(index).zfill(self.zfill_num)), image)

        gt_path = os.path.join(self.gt_root, self.name)
        os.makedirs(gt_path, exist_ok=True)
        imageio.imsave(os.path.join(gt_path, "%s.png" % str(index).zfill(self.zfill_num)), gt_image)

        source_path = os.path.join(self.source_root, self.name)
        os.makedirs(source_path, exist_ok=True)
        imageio.imsave(os.path.join(source_path, "%s.png" % str(index).zfill(self.zfill_num)), source_image)

    def visualize_prediction(self, out, index):
        image = self.visualizer.visualize_out(out)
        pred_path = os.path.join(self.pred_root, self.name)
        os.makedirs(pred_path, exist_ok=True)
        imageio.imsave(os.path.join(pred_path, "%s.png" % str(index).zfill(self.zfill_num)), image)        



class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2  # convert the position from [-1, 1] to real position
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]), self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        # visulize images 
        # Source image+keypoints
        # transformed_frame+keypoints
        # driving image+keypoints
        # Deformed image
        # Prediction
        # Occlusion map

        # 11 sparse deformed:
        # sparse_deformed image0
        # sparse_deformed image0+mask0

        # full_image mask

        images = []

        # Source image with keypoints
        source = source.data.cpu().repeat(1, 3, 1, 1)
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().repeat(1, 3, 1, 1).numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'].data.cpu().numpy()
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        driving = driving.data.cpu().repeat(1, 3, 1, 1).numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().repeat(1, 3, 1, 1).numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        if 'prediction_texture' in out:
            texture = out['prediction_texture'].data.cpu().repeat(1, 3, 1, 1).numpy()
            texture = np.transpose(texture, [0, 2, 3, 1])
            images.append(texture)

        if 'prediction_low' in out:
            low = out['prediction_low'].data.cpu().repeat(1, 3, 1, 1).numpy()
            low = np.transpose(low, [0, 2, 3, 1])
            images.append(low)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().repeat(1, 3, 1, 1).numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

            #彩色遮挡图
            a = occlusion_map
            a = np.array(a)
            a = a*255
            a = a.astype(np.uint8)
            for i in range(a.shape[0]):
                a[i] = cv2.applyColorMap(a[i], cv2.COLORMAP_RAINBOW)
            a = a*1.0/255
            images.append(a)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].data.cpu().repeat(1, 3, 1, 1)
                image = F.interpolate(image, size=source.shape[1:3])
                mask = out['mask'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)

            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
    
    # 画图用
    def visualize_draw(self, driving, source, out):
        images = []

        # Source image
        source = source.data.cpu().repeat(1, 3, 1, 1)
        source = np.transpose(source, [0, 2, 3, 1])
        images.append(source)

        # Source image with unsupervised keypoints
        kp_source = out['kp_source']['value'].data.cpu().numpy()[:, :10, :]
        images.append((source, kp_source))

        # Source image with supervised keypoints
        kp_source = out['kp_source']['value'].data.cpu().numpy()[:, -2:, :]
        images.append((source, kp_source))

        # Source image with all keypoints
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        images.append((source, kp_source))

        # source image, black background, unsupervised keypoints
        bk_img = np.zeros(source.shape, dtype=np.float32)
        kp_source = out['kp_source']['value'].data.cpu().numpy()[:, :10, :]
        images.append((bk_img, kp_source))

        # source image, black background, supervised keypoints
        bk_img = np.zeros(source.shape, dtype=np.float32)
        kp_source = out['kp_source']['value'].data.cpu().numpy()[:, -2:, :]   
        images.append((bk_img, kp_source))

        # source image, black background, all keypoints
        bk_img = np.zeros(source.shape, dtype=np.float32)
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        images.append((bk_img, kp_source))

        # driving image
        driving = driving.data.cpu().repeat(1, 3, 1, 1)
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append(driving)

        # driving image with unsupervised keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()[:, :10, :]
        images.append((driving, kp_driving))

        # driving image with supervised keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()[:, -2:, :]
        images.append((driving, kp_driving))

        # driving image with all keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        images.append((driving, kp_driving))

        # driving image, black background, unsupervised keypoints
        bk_img = np.zeros(driving.shape, dtype=np.float32)
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()[:, :10, :]
        images.append((bk_img, kp_driving))

        # driving image, black background, supervised keypoints
        bk_img = np.zeros(driving.shape, dtype=np.float32)
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()[:, -2:, :] 
        images.append((bk_img, kp_driving))  

        # driving image, black background, all keypoints
        bk_img = np.zeros(driving.shape, dtype=np.float32)
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()  
        images.append((bk_img, kp_driving)) 

        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        if 'prediction_low' in out:
            low = out['prediction_low'].data.cpu().repeat(1, 3, 1, 1).numpy()
            low = np.transpose(low, [0, 2, 3, 1])
            images.append(low)

        if 'prediction_texture' in out:
            texture = out['prediction_texture'].data.cpu().repeat(1, 3, 1, 1).numpy()
            texture = np.transpose(texture, [0, 2, 3, 1])
            images.append(texture)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().repeat(1, 3, 1, 1).numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        if 'deformation' in out:
            image = out['deformation'].data.cpu()
            # image = F.interpolate(image, size=source.shape[1:3])
            np.save("deformation.npy", image)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image

    def visualize_rec(self, driving, source, out):
        # visulize images 
        # Source image+keypoints
        # transformed_frame+keypoints
        # driving image+keypoints
        # Deformed image
        # Prediction
        # Occlusion map

        # 11 sparse deformed:
        # sparse_deformed image0
        # sparse_deformed image0+mask0

        # full_image mask

        images = []

        # Source image with keypoints
        source = source.data.cpu().repeat(1, 3, 1, 1)
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        driving = driving.data.cpu().repeat(1, 3, 1, 1).numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))


        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().repeat(1, 3, 1, 1).numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].data.cpu().repeat(1, 3, 1, 1)
                image = F.interpolate(image, size=source.shape[1:3])
                mask = out['mask'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)

            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image



class Visualizer_demo:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array, flag_sup=False):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        if flag_sup:
            kp_array = kp_array[-2:]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]), self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp, flag_sup=False):
        image_array = np.array([self.draw_image_with_kp(v, k, flag_sup=flag_sup) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, flag_sup=False, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1], flag_sup=flag_sup))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out, flag_sup=False):
        # visulize images 
        # Source image+keypoints
        # transformed_frame+keypoints
        # driving image+keypoints
        # Deformed image
        # Prediction

        images = []

        # Source image with keypoints
        source = source.data.cpu().repeat(1, 3, 1, 1)
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        driving = driving.data.cpu().repeat(1, 3, 1, 1).numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        if 'prediction_texture' in out:
            texture = out['prediction_texture'].data.cpu().repeat(1, 3, 1, 1).numpy()
            texture = np.transpose(texture, [0, 2, 3, 1])
            images.append(texture)

        if 'prediction_low' in out:
            low = out['prediction_low'].data.cpu().repeat(1, 3, 1, 1).numpy()
            low = np.transpose(low, [0, 2, 3, 1])
            images.append(low)

        # Deformed image
        # if 'deformed' in out:
        #     deformed = out['deformed'].data.cpu().numpy()
        #     deformed = np.transpose(deformed, [0, 2, 3, 1])
        #     images.append(deformed)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().repeat(1, 3, 1, 1).numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        image = self.create_image_grid(flag_sup, *images)
        image = (255 * image).astype(np.uint8)
        return image

    def visualize_for_metrics(self, driving, source, out):
        gt_images = []  # driving video
        images = []  # prediction
        source_images = []  # source

        # Driving image with keypoints
        driving = driving.data.cpu().repeat(1, 3, 1, 1).numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        gt_images.append(driving)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().repeat(1, 3, 1, 1).numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        # Source image with keypoints
        source = source.data.cpu().repeat(1, 3, 1, 1)
        source = np.transpose(source, [0, 2, 3, 1])
        source_images.append(source)

        out = []
        for arg in images:
            out.append(self.create_image_column(arg))
        image = np.concatenate(out, axis=1)
        image = (255 * image).astype(np.uint8)

        out = []
        for arg in gt_images:
            out.append(self.create_image_column(arg))
        gt_image = np.concatenate(out, axis=1)
        gt_image = (255 * gt_image).astype(np.uint8)

        out = []
        for arg in source_images:
            out.append(self.create_image_column(arg))
        source_image = np.concatenate(out, axis=1)
        source_image = (255 * source_image).astype(np.uint8)

        return image, gt_image, source_image

    def visualize_out(self, out):
        # gt_images = []  # driving video
        images = []  # prediction

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().repeat(1, 3, 1, 1).numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        out = []
        for arg in images:
            out.append(self.create_image_column(arg))
        image = np.concatenate(out, axis=1)
        image = (255 * image).astype(np.uint8)

        return image
