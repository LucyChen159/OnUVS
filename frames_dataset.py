import os
from skimage import io, img_as_float32, img_as_ubyte
from skimage.color import gray2rgb, rgb2gray
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform, p_aug, img_aug
import glob
import json
from random import sample
import cv2
import imageio
from skimage.transform import resize

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name) and False:  # folder with videos
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif (name.lower().endswith('.png') or name.lower().endswith('.jpg')) and False:  # an image of concatenated frames
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'): # video
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        if video.shape[-1] == 3:  # RGB2GREY
            video = np.array([rgb2gray(frame) for frame in video])
        video = np.expand_dims(video, -1)
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

def read_json(json_path):
    """
    Read json, the format:
      - {'0': {'S': [...], 'U': [...], 'R': [...], 'E': [...], ...},
      - '1': {'S': [...], 'U': [...], 'R': [...], 'E': [...], ...}, ...}
      - the first index is the frame index
    """
    with open(json_path, encoding='utf-8') as f:
        json_labels = json.load(f)
    
    return json_labels


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 1), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.is_train:
            video_array = img_as_ubyte(video_array)
            video_array = img_aug(video_array)
            video_array = img_as_float32(video_array)

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class FramesDatasetWithJson(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, root_json_dir, frame_shape=(256, 256, 1), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.root_json_dir = root_json_dir
        self.videos = os.listdir(root_dir)
        self.jsons = os.listdir(root_json_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
                train_jsons = os.listdir(os.path.join(root_json_dir, 'train'))
                train_videos.sort()
                train_jsons.sort()

            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            test_jsons = os.listdir(os.path.join(root_json_dir, 'test'))
            test_videos.sort()
            test_jsons.sort()
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            self.root_json_dir = os.path.join(self.root_json_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
            self.jsons = train_jsons
        else:
            self.videos = test_videos
            self.jsons = test_jsons

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
            name_json = None
            json_path = None
        else:
            name = self.videos[idx]
            name_json = self.jsons[idx]
            path = os.path.join(self.root_dir, name)
            json_path = os.path.join(self.root_json_dir, name_json)

        video_name = os.path.basename(path)
        json_name = os.path.basename(json_path)

        if self.is_train and os.path.isdir(path) and False:  # images format
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:  # video format
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            json_dict = read_json(json_path)

            selected_index = np.random.choice(num_frames, replace=True, size=2)
            while (json_dict.get('%d'%(selected_index[0])) is None) or (json_dict.get('%d'%(selected_index[1])) is None) \
                or (json_dict['%d'%(selected_index[0])].get('U') is None) or (json_dict['%d'%(selected_index[1])].get('U') is None) \
                or (json_dict['%d'%(selected_index[0])].get('S') is None) or (json_dict['%d'%(selected_index[1])].get('S') is None):
                selected_index = np.random.choice(num_frames, replace=True, size=2)
            frame_idx = selected_index if self.is_train else range(
                num_frames)

            video_array = video_array[frame_idx]
            
            if (json_dict.get('%d'%(frame_idx[0])) is not None) and (json_dict.get('%d'%(frame_idx[1])) is not None):
                U_pos_d = np.float32(json_dict['%d'%(frame_idx[0])]['U'][:-1])
                S_pos_d = np.float32(json_dict['%d'%(frame_idx[0])]['S'][:-1])
                driving_kp = np.stack((U_pos_d, S_pos_d))   # (2, 2)

                U_pos_s = np.float32(json_dict['%d'%(frame_idx[1])]['U'][:-1])
                S_pos_s = np.float32(json_dict['%d'%(frame_idx[1])]['S'][:-1])
                source_kp = np.stack((U_pos_s, S_pos_s))
            else:
                driving_kp = None
                source_kp = None          


        # point augmentation
        if self.is_train:
            video_array = img_as_ubyte(video_array)
            video_array, kp = p_aug(video_array, driving_kp, source_kp)
            driving_kp = kp[0]
            source_kp = kp[1]
            video_array = img_as_float32(video_array)

        if self.transform is not None:
            video_array = self.transform(video_array)
        
        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            if source.shape==(256,256):
                source = np.expand_dims(source, axis=2)
            if driving.shape==(256,256):
                driving = np.expand_dims(driving, axis=2)
                
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))

            # convert to range [-1, 1]
            driving_kp = driving_kp / np.array(self.frame_shape[:2]) * 2 - 1
            source_kp = source_kp / np.array(self.frame_shape[:2]) * 2 - 1

            out['driving_kp'] = driving_kp
            out['source_kp'] = source_kp
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out

class FramesDatasetWithJson_DDH(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, root_json_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        super(FramesDatasetWithJson_DDH, self).__init__()
        self.root_dir = root_dir
        self.root_json_dir = root_json_dir
        self.videos = os.listdir(os.path.join(root_dir,'test'))
        self.jsons = os.listdir(root_json_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, 'test', name)
        print("视频路径：", path)

        video_name = os.path.basename(path)

        
        video_array = read_video(path, frame_shape=self.frame_shape)
        num_frames = len(video_array)
           
        frame_idx = range(num_frames)

        video_array = video_array[frame_idx]
        out = {}
       
        video = np.array(video_array, dtype='float32')
        out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class FramesDatasetWithJson_DDH(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, root_json_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        super(FramesDatasetWithJson_DDH, self).__init__()
        self.root_dir = root_dir
        self.root_json_dir = root_json_dir
        self.videos = os.listdir(root_dir)
        self.jsons = os.listdir(root_json_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
                train_jsons = os.listdir(os.path.join(root_json_dir, 'train'))
                train_videos.sort()
                train_jsons.sort()

            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            test_jsons = os.listdir(os.path.join(root_json_dir, 'test'))
            test_videos.sort()
            test_jsons.sort()
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            self.root_json_dir = os.path.join(self.root_json_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            # train_videos, train_jsons = select_json(train_videos, train_jsons, self.root_json_dir)
            self.videos = train_videos
            self.jsons = train_jsons
        else:
            self.videos = test_videos
            self.jsons = test_jsons

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # cv2.setNumThreads(0)
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
            name_json = None
            json_path = None
        else:
            name = self.videos[idx]
            name_json = self.jsons[idx]
            path = os.path.join(self.root_dir, name)
            json_path = os.path.join(self.root_json_dir, name_json)

        video_name = os.path.basename(path)

         # video format
        video_array = read_video(path, frame_shape=self.frame_shape)
        num_frames = len(video_array)-1
        json_dict = read_json(json_path)

        #修改后的部分
        frame_list = json_dict.keys()
        selected_index = sample(frame_list, 2)
        selected_index = [int(selected_index[0]), int(selected_index[1])] if self.is_train else range(
            num_frames)
        
        nu = 0
     

        frame_idx = selected_index if self.is_train else range(
            num_frames)

        video_array = video_array[frame_idx]
        
        if (json_dict.get('%d'%(frame_idx[0])) is not None) and (json_dict.get('%d'%(frame_idx[1])) is not None) and self.is_train:
            point0_pos_d = np.float32(json_dict['%d'%(frame_idx[0])]['6'][:-1])
            point1_pos_d = np.float32(json_dict['%d'%(frame_idx[0])]['7'][:-1])
            point2_pos_d = np.float32(json_dict['%d'%(frame_idx[0])]['8'][:-1])
            driving_kp = np.stack([point0_pos_d, point1_pos_d, point2_pos_d], axis=0)   # (3, 2)

            point0_pos_s = np.float32(json_dict['%d'%(frame_idx[1])]['6'][:-1])
            point1_pos_s = np.float32(json_dict['%d'%(frame_idx[1])]['7'][:-1])
            point2_pos_s = np.float32(json_dict['%d'%(frame_idx[1])]['8'][:-1])
            source_kp = np.stack([point0_pos_s, point1_pos_s, point2_pos_s], axis=0)   # (2, 2)
            # source_kp = np.stack((point0_pos_s, point1_pos_s))   # (2, 2)
        else:
            driving_kp = None
            source_kp = None          


        # point augmentation
        if self.is_train:
            video_array = img_as_ubyte(video_array)
            video_array, kp = p_aug(video_array, driving_kp, source_kp)
            driving_kp = kp[0]
            source_kp = kp[1]
            video_array = img_as_float32(video_array)

        if self.transform is not None:
            video_array = self.transform(video_array)
        
        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))

            # convert to range [-1, 1]
            driving_kp = driving_kp / np.array(self.frame_shape[:2]) * 2 - 1
            source_kp = source_kp / np.array(self.frame_shape[:2]) * 2 - 1

            out['driving_kp'] = driving_kp
            out['source_kp'] = source_kp
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}


class FramesDatasetWithJson_single(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """
    def __init__(self, root_dir):
        super(FramesDatasetWithJson_single, self).__init__()
        
        self.videos = []
        driving_list = os.listdir(root_dir)
        for single_video in driving_list:
            driving_path = os.path.join(root_dir, single_video)
            path_dict = {}
            path_dict["driving"] = driving_path
            self.videos.append(path_dict)


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):     

        path_dict = self.videos[idx]
        path_d = path_dict["driving"]
        
        name_d = path_d.split('/')[-1].split('.')[0]
        reader = imageio.get_reader(path_d)

        driving_video = []
        try:
            for im in reader:
                im = rgb2gray(im)
                im = np.expand_dims(im, -1)
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [resize(frame, (256, 256)) for frame in driving_video]

        out = {}
        video = np.array(driving_video, dtype='float32')
        out['video'] = video.transpose((3, 0, 1, 2))
        out['d_name'] = name_d

        return out