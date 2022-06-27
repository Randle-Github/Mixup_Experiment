import sys
# sys.path.append('/opt/data/private/DL_Workspace/CompAction/code/')
# sys.path.append('/opt/data/private/DL_Workspace/CompAction/code/model/')
import os
# os.chdir('/opt/data/private/DL_Workspace/CompAction/code')
from os.path import join
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
from data_utils import gtransforms
from .transforms import build_transforms
from .data_parser import WebmDataset
from numpy.random import choice as ch
import time
import json
import cv2
import torch.nn.functional as F


class VideoFolder(torch.utils.data.Dataset):
    """
    @brief      原来 something-else 提供的 Dataset 类

    @details    从 interactive fusoin 那篇的代码里加入了一些修改
    """

    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 frames_duration,
                 args=None,
                 multi_crop_test=False,
                 sample_rate=2,
                 is_test=False,
                 is_val=False,
                 num_boxes=10,
                 model=None,
                 if_augment=True):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param frames_duration: number of frames
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        self.coord_nr_frames = self.in_duration // 2
        self.multi_crop_test = multi_crop_test
        self.sample_rate = sample_rate
        self.if_augment = if_augment
        self.is_val = is_val
        self.data_root = root
        self.dataset_object = WebmDataset(file_input,
                                          file_labels,
                                          root,
                                          is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.model = model
        self.num_boxes = num_boxes

        # Prepare data for the data loader
        self.args = args
        self.prepare_data()
        self.pre_resize_shape = (256, 340)
        self.bbox_folder_path = args.tracked_boxes

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Transformations
        if not self.is_val:
            self.transforms = [
                gtransforms.GroupResize((224, 224)),
            ]
        elif self.multi_crop_test:
            self.transforms = [
                gtransforms.GroupResize((256, 256)),
                gtransforms.GroupRandomCrop((256, 256)),
            ]
        else:
            self.transforms = [
                gtransforms.GroupResize(
                    (224, 224))  # gtransforms.GroupCenterCrop(256),
            ]
        self.transforms += [
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(self.img_mean, self.img_std),
        ]
        self.transforms = Compose(self.transforms)

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(
                    output_size=224, scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(
                    output_size=224,
                    scales=[1],
                    max_distort=0,
                    center_crop_only=True)
        else:
            self.random_crop = None

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        This process will take up a long time, I want to save these objects into a file and read it
        :return:
        """
        self.label_strs = [
            '_'.join(class_name.split(' ')) for class_name in self.classes
        ]  # will this attribute be used?
        vid_names = []
        labels = []
        frame_cnts = []
        # 从 interactive fusion 那里下载的 video_info.json ####################
        # 用来读取每个视频的 size 信息 ########################################
        # video_info_json_pth = '../data/dataset_splits/compositional/video_info.json'
        video_info_json_pth = '../data/dataset_splits/compositional/video_info.json'
        print(os.getcwd())
        print(video_info_json_pth)
        if os.path.exists(video_info_json_pth):
            with open(video_info_json_pth, 'r') as fp:
                self.video_info_json = json.load(fp)
        else:
            self.video_info_json = None

        if not self.is_val:
            with open('../data/train_vid_name.json', 'r') as f:
                vid_names = json.load(f)
            with open('../data/train_labels.json', 'r') as f:
                labels = json.load(f)
            with open('../data/train_frame_cnts.json', 'r') as f:
                frame_cnts = json.load(f)
        else:
            with open('../data/val_vid_name.json', 'r') as f:
                vid_names = json.load(f)
            with open('../data/val_labels.json', 'r') as f:
                labels = json.load(f)
            with open('../data/val_frame_cnts.json', 'r') as f:
                frame_cnts = json.load(f)
        self.vid_names = vid_names
        self.labels = labels
        self.frame_cnts = frame_cnts

    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        file_path = join(os.path.dirname(self.data_root), 'frames', vid_name,
                         '%04d.jpg' % (frame_idx + 1))

        return Image.fromarray(cv2.imread(file_path)).convert('RGB')

    def _sample_indices(self, nr_video_frames):
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                      + np.random.uniform(0, average_duration, size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(
                np.random.randint(nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([
                int(tick / 2.0 + tick * x) for x in range(self.coord_nr_frames)
            ])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        n_frame = self.frame_cnts[index] - 1
        d = self.in_duration * self.sample_rate
        if n_frame > d:
            if not self.is_val:
                # random sample
                offset = np.random.randint(0, n_frame - d)
            else:
                # center crop, 从中间抽样
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            # Temporal Augmentation
            if not self.is_val:  # train
                if n_frame - 2 < self.in_duration:
                    # less frames than needed
                    pos = np.linspace(0, n_frame - 2, self.in_duration)
                else:  # take one
                    pos = np.sort(
                        np.random.choice(list(range(n_frame - 2)),
                                         self.in_duration,
                                         replace=False))
            else:
                pos = np.linspace(0, n_frame - 2, self.in_duration)
            frame_list = [round(p) for p in pos]

        frame_list = [int(x) for x in frame_list]

        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(n_frame)
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)

        # NOTE: 将 coord_frame_list 和 frame_list 设置为一样大
        assert len(coord_frame_list) == len(frame_list) // 2

        folder_id = str(int(self.vid_names[index]))
        video_data = self.load_one_video_json(folder_id)  # 加载 video 的标注 json 信息

        # union the objects of two frames
        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data[
                    'standard_category']  # standard category: 物体0, 物体1
                object_set.add(standard_category)
        object_set = sorted(list(object_set))

        # load frames #########################################################
        frames = []
        if self.model.startswith('coord'):
            pass
        else:
            for fidx in frame_list:
                frames.append(self.load_frame(self.vid_names[index], fidx))

        # resize and crop frames ##############################################
        height, width = self.video_info_json[self.vid_names[index]]['res']
        if frames:
            frames = [
                img.resize((self.pre_resize_shape[1], self.pre_resize_shape[0]),
                           Image.BILINEAR) for img in frames
            ]
        if self.random_crop is not None:
            frames, (offset_h, offset_w, crop_h,
                     crop_w) = self.random_crop(frames)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(
            width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        box_tensors = torch.zeros((self.coord_nr_frames, self.num_boxes, 4),
                                  dtype=torch.float32)  # (cx, cy, w, h)
        box_categories = torch.zeros((self.coord_nr_frames, self.num_boxes))
        for frame_index, frame_id in enumerate(coord_frame_list):
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                global_box_id = object_set.index(standard_category)

                box_coord = box_data['box2d']
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord[
                    'x2'], box_coord['y2']

                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w - 1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h - 1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

                # (cx, cy, w, h)
                gt_box = np.array([(x0 + x1) / 2.,
                                   (y0 + y1) / 2., x1 - x0, y1 - y0],
                                  dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box /= 224

                standard_category = box_data['standard_category']
                # load box into tensor
                try:
                    box_tensors[frame_index,
                                global_box_id] = torch.tensor(gt_box).float()
                except:
                    pass
                try:
                    box_categories[frame_index, global_box_id] = 1 if box_data[
                        'standard_category'] == 'hand' else 2  # 0 is for none
                except:
                    pass

        return frames, box_tensors, box_categories, box_categories, frame_list

    def __getitem__(self, index):
        '''
        '''
        frames, box_tensors, box_categories, frame_ids = self.sample_single(
            index)
        if self.model.startswith('coord'):
            global_img_tensors = []
        else:
            frames = self.transforms(frames)  # original size is (t, c, h, w)
            global_img_tensors = frames.permute(1, 0, 2, 3)  # (c, t, h, w)
        return global_img_tensors, box_tensors, box_categories, self.classes_dict[
            self.labels[index]]

    def __len__(self):
        return len(self.json_data)  # number of videos in train or val

    def unnormalize(self, img, divisor=255):
        """
        The inverse operation of normalization
        Both the input & the output are in the format of BxCxHxW
        """
        for c in range(len(self.img_mean)):
            img[:, c, :, :].mul_(self.img_std[c]).add_(self.img_mean[c])

        return img / divisor


class MyVideoFolder(torch.utils.data.Dataset):
    """
    我自己修改后的 VideoFolder
    """

    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 frames_duration,
                 args=None,
                 multi_crop_test=False,
                 sample_rate=2,
                 is_test=False,
                 is_val=False,
                 num_boxes=10,
                 model=None,
                 if_augment=True):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param frames_duration: number of frames
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        # self.coord_nr_frames = self.in_duration // 2
        self.coord_nr_frames = self.in_duration
        self.multi_crop_test = multi_crop_test
        self.sample_rate = sample_rate
        self.if_augment = if_augment
        self.is_val = is_val
        self.data_root = root
        self.dataset_object = WebmDataset(file_input,
                                          file_labels,
                                          root,
                                          is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.model = model
        self.num_boxes = num_boxes

        # Prepare data for the data loader
        self.args = args
        self.prepare_data()
        self.pre_resize_shape = (256, 340)
        self.bbox_folder_path = args.tracked_boxes

        # Transformations
        self.transforms = build_transforms(not self.is_val)  #  NOTE 能这么写吗这里？

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(
                    output_size=224, scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(
                    output_size=224,
                    scales=[1],
                    max_distort=0,
                    center_crop_only=True)
        else:
            self.random_crop = None

        self.comp_objects_map = None
        with open(
                '../data/dataset_splits/compositional/new_comp_objects_map.json',
                'r') as f:
            self.comp_objects_map = json.load(f)

    def load_one_video_json(self, folder_id):
        while (1):
            try:
                with open(os.path.join(self.bbox_folder_path,
                                       folder_id + '.json'),
                          'r',
                          encoding='utf-8') as f:
                    video_data = json.load(f)
                break
            except IOError:
                print('IOError occur')
                pass
        return video_data

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        This process will take up a long time, I want to save these objects into a file and read it
        :return:
        """
        # print("Loading label strings")
        self.label_strs = [
            '_'.join(class_name.split(' ')) for class_name in self.classes
        ]  # will this attribute be used?
        vid_names = []
        labels = []
        frame_cnts = []
        if not self.is_val:
            with open('../data/train_vid_name.json', 'r') as f:
                vid_names = json.load(f)
            with open('../data/train_labels.json', 'r') as f:
                labels = json.load(f)
            with open('../data/train_frame_cnts.json', 'r') as f:
                frame_cnts = json.load(f)
        else:
            with open('../data/val_vid_name.json', 'r') as f:
                vid_names = json.load(f)
            with open('../data/val_labels.json', 'r') as f:
                labels = json.load(f)
            with open('../data/val_frame_cnts.json', 'r') as f:
                frame_cnts = json.load(f)
        self.vid_names = vid_names
        self.labels = labels
        self.frame_cnts = frame_cnts

    # todo: might consider to replace it to opencv, should be much faster
    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        while (1):
            try:
                img = Image.open(
                    join(os.path.dirname(self.data_root), 'frames', vid_name,
                         '%04d.jpg' % (frame_idx + 1))).convert('RGB')
                break
            except IOError:
                print('IOError occur')
                pass
        return img

    def _sample_indices(self, nr_video_frames):
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                      + np.random.uniform(0, average_duration, size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(
                np.random.randint(nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([
                int(tick / 2.0 + tick * x) for x in range(self.coord_nr_frames)
            ])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        n_frame = self.frame_cnts[index] - 1
        d = self.in_duration * self.sample_rate
        if n_frame > d:
            if not self.is_val:
                # random sample
                offset = np.random.randint(0, n_frame - d)
            else:
                # center crop, 从中间抽样
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            # Temporal Augmentation
            if not self.is_val:  # train
                if n_frame - 2 < self.in_duration:
                    # less frames than needed
                    pos = np.linspace(0, n_frame - 2, self.in_duration)
                else:  # take one
                    pos = np.sort(
                        np.random.choice(list(range(n_frame - 2)),
                                         self.in_duration,
                                         replace=False))
            else:
                pos = np.linspace(0, n_frame - 2, self.in_duration)
            frame_list = [round(p) for p in pos]

        frame_list = [int(x) for x in frame_list]

        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(
                n_frame)  # NOTE: sample_indices 有什么特点
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)

        # NOTE: 将 coord_frame_list 和 frame_list 设置为一样大
        # assert len(coord_frame_list) == len(frame_list) // 2

        folder_id = str(int(self.vid_names[index]))

        # video_data = self.box_annotations[folder_id]
        video_data = self.load_one_video_json(folder_id)

        # union the objects of two frames
        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data[
                    'standard_category']  # standard category: 物体0, 物体1
                if standard_category != 'hand':
                    object_set.add(standard_category)
        object_set = sorted(list(object_set))

        # NOTE: IMPORTANT!!! To accelerate the data loader, here it only reads one image
        #  (they're of the same scale in a video) to get its height and width
        #  It must be modified for models using any appearance features.
        frames = []
        for fidx in coord_frame_list:
            frames.append(self.load_frame(self.vid_names[index], fidx))
            break  # only one image
        height, width = frames[0].height, frames[0].width

        frames = [
            img.resize((self.pre_resize_shape[1], self.pre_resize_shape[0]),
                       Image.BILINEAR) for img in frames
        ]  # just one frame in List:frames

        if self.random_crop is not None:
            frames, (offset_h, offset_w, crop_h,
                     crop_w) = self.random_crop(frames)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape
        if self.model not in [
                'coord', 'coord_latent', 'coord_latent_nl',
                'coord_latent_concat', 'coord_new', 'coord_plus',
                'coord_latent_plus', 'motion', 'shape', 'dynamic',
                'cls_coord_latent'
        ]:
            frames = []
            for fidx in coord_frame_list:
                frames.append(self.load_frame(self.vid_names[index], fidx))
        else:
            # Now for accelerating just pretend we have had frames
            frames = frames * self.in_duration  # TODO:repeat the first loaded frame nr_frames times

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(
            width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        obj_box_tensors = torch.zeros(
            (self.coord_nr_frames, self.num_boxes - 2, 4),
            dtype=torch.float32)  # (cx, cy, w, h)
        hand_box_tensors = torch.zeros((self.coord_nr_frames, 2, 4),
                                       dtype=torch.float32)
        obj_categories_T = torch.zeros(
            (self.coord_nr_frames, self.num_boxes - 2))
        obj_categories = torch.zeros(self.num_boxes)
        box_std_categories = torch.zeros((self.coord_nr_frames, self.num_boxes),
                                         dtype=torch.float32)

        for frame_index, frame_id in enumerate(coord_frame_list):
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            hand_num = 0
            for box_data in frame_data['labels']:

                box_coord = box_data['box2d']
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord[
                    'x2'], box_coord['y2']

                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w - 1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h - 1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

                # (cx, cy, w, h)
                gt_box = np.array([(x0 + x1) / 2.,
                                   (y0 + y1) / 2., x1 - x0, y1 - y0],
                                  dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box /= 224

                standard_category = box_data['standard_category']
                # load box into tensor
                if standard_category == 'hand':
                    try:
                        hand_box_tensors[frame_index, hand_num] = torch.tensor(
                            gt_box).float()
                        box_std_categories[frame_index, self.num_boxes - 2 +
                                           hand_num] = 1  # 1 denotes hand
                        obj_categories[self.num_boxes - 2 +
                                       hand_num] = 372  # 把 hand 也视作一个 object 类别
                        hand_num += 1
                    except IndexError:
                        pass
                else:
                    try:
                        global_box_id = object_set.index(standard_category)
                        if global_box_id < self.num_boxes - 2:
                            obj_box_tensors[frame_index,
                                            global_box_id] = torch.tensor(
                                                gt_box).float()
                            box_std_categories[frame_index, global_box_id] = 2
                            # 加入 object category ############################
                            if self.comp_objects_map is not None:
                                try:
                                    obj_categories_T[
                                        frame_index,
                                        global_box_id] = self.comp_objects_map[
                                            box_data['category']] + 1
                                    if obj_categories[global_box_id] == 0:
                                        obj_categories[
                                            global_box_id] = self.comp_objects_map[
                                                box_data['category']] + 1
                                except KeyError:
                                    pass
                    except IndexError:
                        pass
                # x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))  # region of interest?
        box_tensors = torch.cat([obj_box_tensors, hand_box_tensors],
                                dim=1)  # 最后两个是 hand box tensors

        return frames, box_tensors, box_std_categories, obj_categories, obj_categories_T

    def __getitem__(self, index):
        '''
        box_tensors: [nr_frames, num_boxes, 4]
        box_categories: [nr_frames, num_boxes], value is 0(none), 1 (hand), 2 (object)
        frames: what about the frames shape?
        '''
        frames, box_tensors, box_categories, obj_categories, obj_categories_T = self.sample_single(
            index)
        frames = self.transforms(frames)  # original size is (t, c, h, w)
        global_img_tensors = frames.permute(1, 0, 2, 3)  # (c, t, h, w)
        return global_img_tensors, box_tensors, box_categories, obj_categories, obj_categories_T, self.classes_dict[
            self.labels[index]]

    def __len__(self):
        return len(self.json_data)

    def unnormalize(self, img, divisor=255):
        """
        The inverse operation of normalization
        Both the input & the output are in the format of BxCxHxW
        """
        for c in range(len(self.img_mean)):
            img[:, c, :, :].mul_(self.img_std[c]).add_(self.img_mean[c])

        return img / divisor

    def img2np(self, img):
        """
        Convert image in torch tensors of BxCxTxHxW [float32] to a numpy array of BxHxWxC [0-255, uint8]
        Take the first frame along temporal dimension
        if C == 1, that dimension is removed
        """
        img = self.unnormalize(img[:, :, 0, :, :],
                               divisor=1).to(torch.uint8).permute(0, 2, 3, 1)
        if img.shape[3] == 1:
            img = img.squeeze(3)
        return img.cpu().numpy()


class InteractionVideoFolder(torch.utils.data.Dataset):
    """
    @brief      针对 subject-object interation graph 设计的 dataset

    @details    detailed description
    """

    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 frames_duration,
                 args=None,
                 multi_crop_test=False,
                 sample_rate=2,
                 is_test=False,
                 is_val=False,
                 num_boxes=4,
                 model=None,
                 if_augment=True):
        ""
        self.in_duration = frames_duration
        self.coord_nr_frames = self.in_duration // 2
        self.multi_crop_test = multi_crop_test
        self.sample_rate = sample_rate
        self.if_augment = if_augment
        self.is_val = is_val
        self.data_root = root
        self.dataset_object = WebmDataset(file_input,
                                          file_labels,
                                          root,
                                          is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.model = model
        self.num_boxes = num_boxes

        # Prepare data for the data loader
        self.args = args
        self.prepare_data()
        self.pre_resize_shape = (256, 340)
        self.bbox_folder_path = args.tracked_boxes

        self.transforms = build_transforms(not self.is_val)

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(
                    output_size=224, scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(
                    output_size=224,
                    scales=[1],
                    max_distort=0,
                    center_crop_only=True)
        else:
            self.random_crop = None

        self.comp_objects_map = None
        with open(
                '../data/dataset_splits/compositional/new_comp_objects_map.json',
                'r') as f:
            self.comp_objects_map = json.load(f)

    def load_one_video_json(self, video_id):
        while (1):
            try:
                with open(
                        os.path.join(self.bbox_folder_path, video_id + '.json'),
                        'r') as f:
                    video_anno_data = json.load(f)
                    break
            except IOError:
                print('IOError occur')
                pass
        return video_anno_data

    def prepare_data(self):
        # self.vid_info
        # self.vid_names
        # self.vide_labels
        # self.frame_cnqts
        video_info_json_pth = '../data/dataset_splits/compositional/video_info.json'
        # video_info_json_pth = '/opt/data/private/DL_Workspace/CompAction/data/dataset_splits/compositional/video_info.json'
        if os.path.exists(video_info_json_pth):
            with open(video_info_json_pth, 'r') as fp:
                self.vid_info = json.load(fp)
        else:
            self.vid_info = None
        if self.is_val:
            with open('../data/val_vid_name.json', 'r') as f:
                self.vid_names = json.load(f)
            with open('../data/val_labels.json', 'r') as f:
                self.labels = json.load(f)
            with open('../data/val_frame_cnts.json', 'r') as f:
                self.frame_cnts = json.load(f)
        else:
            with open('../data/train_vid_name.json', 'r') as f:
                self.vid_names = json.load(f)
            with open('../data/train_labels.json', 'r') as f:
                self.labels = json.load(f)
            with open('../data/train_frame_cnts.json', 'r') as f:
                self.frame_cnts = json.load(f)

    def load_frame(self, vid_name, frame_idx):
        file_path = join(os.path.dirname(self.data_root), 'frames', vid_name,
                         '%04d.jpg' % (frame_idx + 1))

        return Image.fromarray(cv2.imread(file_path)).convert('RGB')

    def _sample_indices(self, nr_video_frames):
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                + np.random.uniform(0, average_duration, size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        # 什么情况下会到这个条件里面？ ####################################
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(
                np.random.randint(nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    #  NOTE 和上面这种 indices sampler 有什么区别
    def _get_val_indices(self, nr_video_frames):
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([
                int(tick / 2.0 + tick * x) for x in range(self.coord_nr_frames)
            ])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def sample_single(self, index):
        n_frame = self.frame_cnts[index] - 1
        # 下面这一段得到 frame index 的代码也可以重用 #####################
        d = self.in_duration * self.sample_rate  # in_duration 是我们要采样的帧数，sample_rate 是采样率
        if n_frame > d:
            if not self.is_val:
                offset = np.random.randint(0, n_frame - d)
            else:
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            if not self.is_val:
                if n_frame - 2 < self.in_duration:
                    pos = np.linspace(
                        0, n_frame - 2,
                        self.in_duration)  #  NOTE 这里会有重复的 frame index 吗
                else:
                    pos = np.sort(
                        np.random.choice(list(range(n_frame - 2)),
                                         self.in_duration,
                                         replace=False))  #  NOTE 这里不会有重复？
            else:
                pos = np.linspace(0, n_frame - 2, self.in_duration)
            frame_list = [round(p) for p in pos]

        frame_list = [int(x) for x in frame_list]

        # coordinate 对应的帧和 frame 的帧不一致 ##########################
        # 这两种方法采出来的 frame 差异有多大 #############################
        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(n_frame)
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)

        assert len(coord_frame_list) == len(frame_list) // 2

        folder_id = str(int(self.vid_names[index]))
        video_anno_data = self.load_one_video_json(folder_id)

        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_anno_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data[
                    'standard_category']  # standard category: 物体0, 物体1
                if standard_category != 'hand':  # 跳过 hand
                    object_set.add(standard_category)
        object_set = sorted(list(object_set))

        # load frames #####################################################
        frames = []
        if self.model.startswith('coord'):
            pass
        else:
            for fidx in frame_list:
                frames.append(self.load_frame(self.vid_names[index], fidx))
        # pre-resize and crop frames ######################################
        height, width = self.vid_info[self.vid_names[index]]['res']
        if frames:
            frames = [
                img.resize((self.pre_resize_shape[1], self.pre_resize_shape[0]),
                           Image.BILINEAR) for img in frames
            ]
        if self.random_crop is not None:
            frames, (offset_h, offset_w, crop_h,
                     crop_w) = self.random_crop(frames)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(
            width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        adjust_bbox_params = {
            'scale_resize_w': scale_resize_w,
            'scale_resize_h': scale_resize_h,
            'offset_w': offset_w,
            'offset_h': offset_h,
            'scale_crop_w': scale_crop_w,
            'scale_crop_h': scale_crop_h,
            'crop_w': crop_w,
            'crop_h': crop_h
        }

        # process bounding boxes ##########################################
        blobs = {}
        blobs['object_boxes'] = torch.zeros(
            (self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32)
        blobs['hand_boxes'] = torch.zeros((self.coord_nr_frames, 1, 4),
                                          dtype=torch.float32)
        blobs['geometric'] = torch.zeros(
            (self.coord_nr_frames, self.num_boxes, 4),
            dtype=torch.float32)  # geometric relation between sub-obj pair
        blobs['obj_indicator'] = torch.zeros(
            (self.num_boxes),
            dtype=torch.long)  # indicate the box is object or not
        # identity 用于表示每帧的中是否出现了 hand 或者 object
        blobs['obj_identity'] = torch.zeros(
            (self.coord_nr_frames, self.num_boxes))
        blobs['hand_identity'] = torch.zeros((self.coord_nr_frames))

        for idx, frame_id in enumerate(coord_frame_list):
            try:
                frame_anno = video_anno_data[frame_id]
            except:
                frame_anno = {'labels': []}

            frame_blobs = {
                'obj': [(0, 0, 0, 0) for i in range(self.num_boxes)],
                'hand': [(0, 0, 0, 0), (0, 0, 0, 0)],
                'geometric': np.zeros((self.num_boxes, 4)),
            }
            hand_num = 0
            for box_data in frame_anno['labels']:
                box = adjust_bbox(box_data['box2d'],
                                  adjust_bbox_params)  # (x1, y1, x2, y2)

                box_category = box_data['standard_category']

                if box_category != 'hand':
                    global_box_id = object_set.index(
                        box_data['standard_category'])
                    try:
                        frame_blobs['obj'][global_box_id] = box
                        blobs['obj_indicator'][global_box_id] = 1
                        blobs['obj_identity'][idx, global_box_id] = 1
                    except:
                        pass
                else:
                    if hand_num < 2:  # process two hands only
                        frame_blobs['hand'][hand_num] = box
                        blobs['hand_identity'][idx] = 1  # 表示在这一帧是有手的
                        hand_num += 1

            # get union of hand box
            hand_box = bbox_union(frame_blobs['hand'][0],
                                  frame_blobs['hand'][1])

            for i in range(self.num_boxes):
                frame_blobs['geometric'][i] = np.array(self.generate_geometric(
                    hand_box, frame_blobs['obj'][i]),
                                                       dtype=np.float32) / 224
                # tranform bbox mode -> xywh
                # nornalize values into [0, 1]
                frame_blobs['obj'][i] = np.array(bbox_trans_mode(
                    frame_blobs['obj'][i]),
                                                 dtype=np.float32)
                frame_blobs['obj'][i] /= 224

            hand_box = np.array(bbox_trans_mode(hand_box), dtype=np.float32)
            hand_box /= 224

            # load into blobs
            blobs['object_boxes'][idx] = torch.tensor(
                frame_blobs['obj']).float()  # t,n,4
            blobs['hand_boxes'][idx,
                                0] = torch.tensor(hand_box).float()  # t,1,4
            blobs['geometric'][idx] = torch.tensor(
                frame_blobs['geometric']).float()  # t,n,4
        # 对 hand 得到它自身的运动轨迹的特征
        blobs['hand_move'] = self.generate_self_move(blobs['hand_boxes'])

        return frames, blobs

    def __getitem__(self, index):
        frames, blobs = self.sample_single(index)
        if self.model.startswith('coord'):
            global_img_tensors = []
        else:
            frames = self.transforms(frames)
            global_img_tensors = frames.permute(1, 0, 2, 3)  # (c, t, h, w)
        label = self.classes_dict[self.labels[index]]
        return global_img_tensors, blobs, label

    def __len__(self):
        return len(self.json_data)

    def generate_self_move(self, boxes):
        # boxes: [t, 1, 4]
        boxes_input = boxes.transpose(0, 1).contiguous()  # [1, t, 4]
        shift_box_input_l, _ = torch.split(boxes_input,
                                           [self.coord_nr_frames - 1, 1],
                                           dim=-2)
        _, shift_box_input_r = torch.split(boxes_input,
                                           [1, self.coord_nr_frames - 1],
                                           dim=-2)
        diff_box_input = shift_box_input_r - shift_box_input_l  # (n, t-1, 4)
        diff_box_input_pluszero = F.pad(diff_box_input,
                                        (0, 0, 1, 0))  #(n, t, 4)
        return diff_box_input_pluszero.transpose(0, 1).contiguous()  # (t, n, 4)

    def generate_geometric(self, hand_box_ori, object_box_ori):
        h = list(hand_box_ori).copy()
        o = list(object_box_ori).copy()
        inter = [h[0] - o[0], h[1] - o[1], h[2] - o[2], h[3] - o[3]]
        # return h + o + inter
        return inter


def adjust_bbox(box, adj_params):
    # box: (x, y, h, w)
    x0, y0, x1, y1 = box['x1'], box['y1'], box['x2'], box['y2']

    # scaling due to initial resize
    x0, x1 = x0 * adj_params['scale_resize_w'], x1 * adj_params['scale_resize_w']
    y0, y1 = y0 * adj_params['scale_resize_h'], y1 * adj_params['scale_resize_h']

    # shift
    x0, x1 = x0 - adj_params['offset_w'], x1 - adj_params['offset_w']
    y0, y1 = y0 - adj_params['offset_h'], y1 - adj_params['offset_h']

    x0, x1 = np.clip([x0, x1], a_min=0, a_max=adj_params['crop_w'] - 1)
    y0, y1 = np.clip([y0, y1], a_min=0, a_max=adj_params['crop_h'] - 1)

    # scaling due to crop
    x0, x1 = x0 * adj_params['scale_crop_w'], x1 * adj_params['scale_crop_w']
    y0, y1 = y0 * adj_params['scale_crop_h'], y1 * adj_params['scale_crop_h']

    # precaution
    x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
    y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

    return (x0, y0, x1, y1)


def bbox_trans_mode(box, mode='xywh'):
    if mode == 'xywh':
        x0, y0, x1, y1 = box
        return [(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0]
    else:  # xywh
        x0, y0, w, h = box
        return [(x0 - w / 2.), (y0 - w / 2.), (x0 + w / 2.), (y0 + w / 2.)]


def bbox_union(boxA, boxB):
    # boxA,B: (x1, y1, x2, y2)
    if bbox_valid_p(boxA) and bbox_valid_p(boxB):
        return [
            min(boxA[0], boxB[0]),
            min(boxA[1], boxB[1]),
            max(boxA[2], boxB[2]),
            max(boxA[3], boxB[3])
        ]
    elif bbox_valid_p(boxA):
        return boxA
    elif bbox_valid_p(boxB):
        return boxB
    else:
        return [0, 0, 0, 0]


def bbox_valid_p(box):
    if box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0:
        return False
    else:
        return True


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict({
        'root_frames': '/opt/data/private/FSL_Datasets/sth-sth-v2/',
        'num_boxes': 4,
        'json_data_train': '../data/dataset_splits/compositional/train.json',
        'json_file_labels': '../data/dataset_splits/compositional/labels.json',
        'num_frames': 16,
        'tracked_boxes': '/opt/data/private/FSL_Datasets/bounding_box_smthsmth',
        'model': 'interactiongraph'
    })

    dataset = InteractionVideoFolder(
        root=args.root_frames,
        num_boxes=args.num_boxes,
        file_input=args.json_data_train,
        file_labels=args.json_file_labels,
        frames_duration=args.num_frames,
        args=args,
        is_val=False,
        if_augment=True,
        model=args.model,
    )

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)

    # for data in data_loader:
    #     print(data[0].size(), data[1]['union_boxes'][0][0][0])
    #     break
    dataset[0]
