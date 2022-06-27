#!/usr/bin/env python3
# 在原来的基础上仿照 pyslowfast 写的 sth-else dataset.

import json
import numpy as np
import os
import os.path as osp
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
class Sthelsebox(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, num_retries=10) -> None:
        self.cfg = cfg
        assert mode in ['train', 'val', 'test'], f"SthElse doesnot support {mode} split"
        self.mode = mode
        assert isinstance(self.cfg.DATA.MEAN, list) and len(self.cfg.DATA.MEAN) == 3, "图片均值设置错误"

        # 保证在不同机器上，路径是都存在的
        self.cfg.DATA.PATH_TO_DATA_DIR = utils.ensure_home_path(cfg.DATA.PATH_TO_DATA_DIR)
        self.pre_resize_shape = (256, 340)
        self.bbox_annotation_path = osp.join(self.cfg.DATA.PATH_TO_DATA_DIR, self.cfg.DATA.BOUNDING_BOX_NAME)
        anno_obj_map_path = osp.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'object_relevant', 'obj_desc_map.json')
        with open(anno_obj_map_path, 'r') as f:
            self.anno_obj_map = json.load(f)

        # 每个 video 的 name, path, frame numbers
        # 每个 list 的长度就对应着 train/val 的样本数目
        self.video_info_dir = osp.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'splits', self.cfg.DATA.STHELSE_SPLIT)
        with open(osp.join(self.video_info_dir, f'{mode}_vid_name.json'), 'r') as f:
            self.vid_names = json.load(f)
        with open(osp.join(self.video_info_dir, f'{mode}_vid_label.json'), 'r') as f:
            self.vid_labels = json.load(f)
        with open(osp.join(self.video_info_dir, f'{mode}_vid_frame_cnts.json'), 'r') as f:
            self.vid_frame_cnts = json.load(f)
        assert len({len(self.vid_names), len(self.vid_labels), len(self.vid_frame_cnts)}) == 1, "json 文件的样本数量不一致"
            

        # Transformation
        # 用的以前的 transform
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
        "建立 label name 到 int 的映射"
        # Load dataset label names
        label_file = osp.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'splits', self.cfg.DATA.STHELSE_SPLIT, 'labels.json')
        with open(label_file, "r") as f:
            label_dict = json.load(f)
        self.classes_dict = {}
        for label, idx in label_dict.items():
            self.classes_dict[label] = idx
            self.classes_dict[idx] = label
        

    def load_anno_json(self, folder_id):
        "加载 video 的 bbox 等标注信息"
        with open(osp.join(self.bbox_annotation_path, folder_id + '.json'),
                  'r',
                  encoding='utf-8') as f:
            video_data = json.load(f)
        return video_data

    def load_frame(self, vid_name, frame_idx):
        file_path = osp.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'frames', vid_name,
                         '%04d.jpg' % (frame_idx + 1))
        return Image.fromarray(cv2.imread(file_path)).convert('RGB')

    def get_seq_frames(self, index):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = self.vid_frame_cnts[index]

        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return seq

    def sample_single(self, index):
        frame_list = self.get_seq_frames(index)
        video_name = self.vid_names[index]
        video_anno_data = self.load_anno_json(video_name)

        object_set = set()
        for fidx in frame_list:
            try:
                frame_anno = video_anno_data[fidx]
            except:
                frame_anno = {'labels': []}
            for box_anno in frame_anno['labels']:
                standard_category = box_anno['standard_category']
                if standard_category == 'hand':
                    continue
                object_set.add(standard_category)
        object_set = sorted(list(object_set))

        # load frames #########################################################
        # frames = []
        frame = self.load_frame(self.vid_names[index], 0)
        height, width = frame.height, frame.width
        offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape
        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(
            width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        # process bounding boxes ##########################################
        # hand box 放到前面 2 个位置，之后再放 object boxes
        meta = {}
        meta['object_boxes'] = torch.zeros(
            (self.cfg.DATA.NUM_FRAMES, self.cfg.DATA.NUM_BOXES, 4), dtype=torch.float32)
        meta['box_category'] = torch.zeros(
            (self.cfg.DATA.NUM_FRAMES, self.cfg.DATA.NUM_BOXES))
        meta['obj_indicator'] = torch.zeros(
            (self.cfg.DATA.NUM_BOXES, ))
        meta['obj_category'] = torch.zeros(
            (self.cfg.DATA.NUM_BOXES, ), dtype=torch.long)

        for i, fidx in enumerate(frame_list):
            try:
                frame_anno = video_anno_data[fidx]
            except:
                frame_anno = {'labels': []}
            hand_cnt = 0
            for box_anno in frame_anno['labels']:
                standard_category = box_anno['standard_category']
                if standard_category == 'hand':
                    if hand_cnt == 0:
                        global_box_id = 0
                        hand_cnt += 1
                    else:
                        global_box_id = 1
                else:
                    global_box_id = object_set.index(standard_category) + 2

                box_coord = box_anno['box2d']
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

                # load box into tensor
                try:
                    meta['object_boxes'][i,
                                          global_box_id] = torch.tensor(
                                              gt_box).float()
                    meta['box_category'][
                        i, global_box_id] = 1 if box_anno[
                            'standard_category'] == 'hand' else 2
                    if box_anno['standard_category'] != 'hand':
                        meta['obj_indicator'][global_box_id] = 1
                        obj_desc = box_anno['category']
                        meta['obj_category'][global_box_id] = self.anno_obj_map[obj_desc]['id'] + 1
                except IndexError:
                    pass
        return frame, meta

    def __getitem__(self, index):
        _, meta = self.sample_single(index)
        # frames = self.transforms(frames)
        # T C H W -> C T H W
        # frames = frames.permute(1, 0, 2, 3)
        # frames = utils.pack_pathway_output(self.cfg, frames)
        dummy_tensor = torch.randn((3, self.cfg.DATA.NUM_FRAMES, 1, 1))
        label = self.classes_dict[self.vid_labels[index]]
        return dummy_tensor, label, index, meta

    def __len__(self):
        return len(self.vid_names)
