#!/usr/bin/env python3
# action genome 的数据加载

import json
import numpy as np
import os
from os.path import join
import random
import torch
import torch.utils.data
from torchvision import transforms
import cv2
from PIL import Image

import compaction.utils.logging as logging

from . import gtransforms as gtransforms
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Actiongenome(torch.utils.data.Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()
        self.cfg = cfg
        assert mode in ['train', 'val'], f'Action Genome does not support {mode} split'
        self.mode = mode
        if mode == 'train':
            self.is_val = False
        else:
            self.is_val = True
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.sample_rate = 2
        # 保证在不同机器上，路径是都存在的
        self.cfg.DATA.PATH_TO_DATA_DIR = utils.ensure_home_path(cfg.DATA.PATH_TO_DATA_DIR)
        self.pre_resize_shape = (256, 340)

        self.transforms = [gtransforms.GroupResize((224, 224)),
                           gtransforms.ToTensor(),
                           gtransforms.GroupNormalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD)]
        self.transforms = transforms.Compose(self.transforms)

        if self.mode == 'train':
            self.random_crop = gtransforms.GroupMultiScaleCrop(
                output_size=224, scales=[1, .875, .75]
            ) # 会随机从这几个 scale 中选择一个进行 crop
        else:
            self.random_crop = gtransforms.GroupMultiScaleCrop(
                    output_size=224,
                    scales=[1],
                    max_distort=0,
                    center_crop_only=True)
        self._construct_loader()

    def _construct_loader(self):
        video_info_file = join(self.cfg.DATA.PATH_TO_DATA_DIR, 'splits', 'video_info.json')
        split_file = join(self.cfg.DATA.PATH_TO_DATA_DIR, 'splits', self.mode+'.json')
        anno_file = join(self.cfg.DATA.PATH_TO_DATA_DIR, 'annotations', 'bounding_box_annotations.json')

        self.video_info_json = utils.load_json_from_file(video_info_file)
        self.video_list = utils.load_json_from_file(split_file)
        self.box_annotations = utils.load_json_from_file(anno_file)

        vid_names = []
        labels = []
        frame_cnts = []
        frame_ids = []

        for elem in self.video_list:
            labels.append(elem['label'])
            vid_names.append(elem['id'])
            frame_cnts.append(self.video_info_json[elem['id']]['cnt'])
            frame_ids.append(self.video_info_json[elem['id']]['frame_ids'])

        self.vid_names = vid_names
        self.labels = labels
        self.frame_cnts = frame_cnts
        self.frame_ids = frame_ids

        # load label
        label_dict = utils.load_json_from_file(join(self.cfg.DATA.PATH_TO_DATA_DIR, 'splits', 'labels.json'))
        self.classes_dict = {}
        for label, idx in label_dict.items():
            self.classes_dict[label] = idx
            self.classes_dict[idx] = label

    def _sample_indices(self, nr_video_frames):
        # random sampling
        average_duration = nr_video_frames * 1.0 / self.cfg.DATA.NUM_FRAMES
        if average_duration > 0:
            offsets = np.multiply(list(range(self.cfg.DATA.NUM_FRAMES)), average_duration) \
                + np.random.uniform(0, average_duration,
                                    size=self.cfg.DATA.NUM_FRAMES)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.cfg.DATA.NUM_FRAMES:
            offsets = np.sort(np.random.randint(
                nr_video_frames, size=self.cfg.DATA.NUM_FRAMES))
        else:
            offsets = np.zeros((self.cfg.DATA.NUM_FRAMES,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        # fixed sampling
        if nr_video_frames > self.cfg.DATA.NUM_FRAMES:
            tick = nr_video_frames * 1.0 / self.cfg.DATA.NUM_FRAMES
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.cfg.DATA.NUM_FRAMES)])
        else:
            offsets = np.zeros((self.cfg.DATA.NUM_FRAMES,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def load_frames(self, vid_name, frame_idx):
        file_path = join(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,
                                      'frames', vid_name, '%s' % (frame_idx)))
        return Image.fromarray(cv2.imread(file_path)).convert('RGB')

    def sample_single(self, index):
        n_frame = self.frame_cnts[index] - 1
        frame_list = self._sample_indices(n_frame)

        d = self.cfg.DATA.NUM_FRAMES * self.sample_rate  # 16 * 2
        if n_frame > d:
            if not self.is_val:
                # random sample
                offset = np.random.randint(0, n_frame - d)
            else:
                # center crop
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            # Temporal Augmentation
            if not self.is_val:  # train
                if n_frame - 2 < self.cfg.DATA.NUM_FRAMES:
                    # less frames than needed
                    pos = np.linspace(0, n_frame - 2, self.cfg.DATA.NUM_FRAMES)
                else:  # take one
                    pos = np.sort(np.random.choice(
                        list(range(n_frame - 2)), self.cfg.DATA.NUM_FRAMES, replace=False))
            else:
                pos = np.linspace(0, n_frame - 2, self.cfg.DATA.NUM_FRAMES)
            frame_list = [round(p) for p in pos]

        folder_id = str(self.vid_names[index])
        video_data = self.box_annotations[folder_id]

        object_set = set()
        for frame_id in frame_list:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                object_set.add(standard_category)
        object_set = sorted(list(object_set))

        frames = []
        for fidx in frame_list:
            fidx = self.frame_ids[index][fidx]
            frames.append(self.load_frames(self.vid_names[index], fidx))

        if self.video_info_json:
            height, width = self.video_info_json[self.vid_names[index]]['res']
        else:
            height, width = frames[0].height, frames[0].width

        # pre-resize imgs
        frames = [
            img.resize((self.pre_resize_shape[1], self.pre_resize_shape[0]),
                       Image.BILINEAR) for img in frames
        ]

        frames, (offset_h, offset_w, crop_h,
                     crop_w) = self.random_crop(frames)

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(
            width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        # load bbox
        meta = {}
        meta['object_boxes'] = torch.zeros(
            (self.cfg.DATA.NUM_FRAMES, self.cfg.DATA.NUM_BOXES, 4), dtype=torch.float32
        )
        meta['box_category'] = torch.zeros(
            (self.cfg.DATA.NUM_FRAMES, self.cfg.DATA.NUM_BOXES), dtype=torch.float32
        )
        for i, frame_id in enumerate(frame_list):
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                global_box_id = object_set.index(standard_category)

                box_coord = box_data['box2d']
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord['x2'], box_coord['y2']

                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w-1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h-1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

                # (cx, cy, w, h)
                gt_box = np.array(
                    [(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box /= 224

                # load box into tensor
                try:
                    meta['object_boxes'][i, global_box_id] = torch.tensor(
                        gt_box).float()
                except:
                    pass
                # load box category #### need a try ... except ... ? Mr. Yan
                try:
                    # box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] == 'hand' else 2  # 0 is for none
                    meta['box_category'][i, global_box_id] = 1 if box_data['standard_category'] in [
                    'hand', 'person'] else 2  # 0 is for none
                except:
                    pass

        return frames, meta

    def __getitem__(self, index):
        frames, meta = self.sample_single(index)
        frames = self.transforms(frames)
        # T C H W -> C T H W
        frames = frames.permute(1, 0, 2, 3)
        frames = utils.pack_pathway_output(self.cfg, frames)
        classes = []
        for label in self.labels[index]:
            classes.append(self.classes_dict[label])
        classes = torch.as_tensor(
            utils.as_binary_vector(classes, self.num_classes))
        return frames, classes, index, meta

    def __len__(self):
        return len(self.vid_names)

