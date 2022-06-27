import os
from os.path import join
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import torch
import time
from data_utils import gtransforms
from data_utils.data_parser import WebmDataset
from numpy.random import choice as ch
import random
import json
import cfg_init
import cv2

model_args = cfg_init.model_args()

class VideoFolder(torch.utils.data.Dataset):
    """
    Something-Something dataset based on *frames* extraction
    这个数据集有一个问题，它采集 frame 的 index 和采集 bounding box 的 index 不一致，并且数量上， bounding box 的数目是 frame 的 1/2.
    即8帧的图片只采集了4帧的 bounding box
    """

    bbox_folder_path = '/mnt/data1/home/sunpengzhan/something-else-idea/bounding_box_smthsmth'

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
        :param sample_split: how many frames sub-sample from each clip
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        self.coord_nr_frames = self.in_duration  # 这里不变成一半
        # self.coord_nr_frames = self.in_duration
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
        # boxes_path = args.tracked_boxes
        # print('... Loading box annotations might take a minute ...')
        # with open(boxes_path, 'r') as f:
        #     self.box_annotations = json.load(f)
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
                gtransforms.GroupResize((224, 224))
                # gtransforms.GroupCenterCrop(256),
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

    def load_one_video_json(self, folder_id):
        with open(os.path.join(self.bbox_folder_path, folder_id + '.json'),
                  'r',
                  encoding='utf-8') as f:
            video_data = json.load(f)
        return video_data

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        This process will take up a long time, I want to save these objects into a file and read it
        :return:
        """
        print("Loading label strings")
        video_info_json_pth = '../data/dataset_splits/compositional/video_info.json'
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

        with open('../data/dataset_splits/compositional/object_relevant/anno_obj_map.json', 'r') as f:
            self.anno_obj_map = json.load(f)

    # todo: might consider to replace it to opencv, should be much faster
    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        # return Image.open(
            # join(os.path.dirname(self.data_root), 'frames', vid_name,
                 # '%04d.jpg' % (frame_idx + 1))).convert('RGB')
        file_path = join(os.path.dirname(self.data_root), 'frames', vid_name,
                         '%04d.jpg' % (frame_idx + 1))
        return Image.fromarray(cv2.imread(file_path)).convert('RGB')

    def _sample_indices(self, nr_video_frames):
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames  # 这个 coord_nr_frames 是什么意思
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
        d = self.in_duration * self.sample_rate  # 16 * 2, sample_rate is the step size, and d is the sample interval, in_duration is the sample length
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
                if n_frame - 2 < self.in_duration:  # why n_frame-2???
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

        # assert len(coord_frame_list) == len(frame_list) // 2
        # assert len(coord_frame_list) == len(frame_list)

        folder_id = str(int(self.vid_names[index]))  # lxs1
        video_anno_data = self.load_one_video_json(folder_id)
        # video_data = self.load_one_video_json(folder_id)
        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_anno_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data[
                    'standard_category']  # standard category: [0001, 0002]
                object_set.add(standard_category)
        object_set = sorted(list(object_set))[::-1]  # 让 hand 始终在最前面的位置

        # load frames #########################################################
        frames = []
        for fidx in frame_list:
            frames.append(self.load_frame(self.vid_names[index], fidx))
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

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(
            width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        # process bounding boxes ##########################################
        blobs = {}
        blobs['object_boxes'] = torch.zeros(
            (self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32)
        blobs['box_category'] = torch.zeros(
            (self.coord_nr_frames, self.num_boxes))
        # 用于指示框是不是 hand
        blobs['hand_identity'] = torch.zeros((self.num_boxes), dtype=torch.long)
        # 指示 object 具体是什么类别
        blobs['object_category'] = torch.full((self.num_boxes, ), 0, dtype=torch.long)

        for frame_index, frame_id in enumerate(coord_frame_list):
            try:
                frame_data = video_anno_data[frame_id]
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

                # load box into tensor
                try:
                    blobs['object_boxes'][frame_index,
                                          global_box_id] = torch.tensor(
                                              gt_box).float()
                    blobs['box_category'][
                        frame_index, global_box_id] = 1 if box_data[
                            'standard_category'] == 'hand' else 2
                    blobs['hand_identity'][global_box_id] = 1 if box_data[
                        'standard_category'] == 'hand' else 2
                except IndexError:
                    pass
                if 'hand' not in standard_category:
                    obj_cat = box_data['category']
                    blobs['object_category'][global_box_id] = self.anno_obj_map[obj_cat]+1  # 将标注的 object 映射成 1-371 的一个数值, 0 保留，表示没有物体
            
        blobs['object_boxes'] = blobs['object_boxes'].view(self.coord_nr_frames, self.num_boxes*4).transpose(0, 1)  # (20, T)
                    
        return frames, blobs

    def __getitem__(self, index):
        '''
        box_tensors: [nr_frames, num_boxes, 4]
        box_categories: [nr_frames, num_boxes], value is 0(none), 1 (hand), 2 (object)
        frames: what about the frames shape?
        '''
        frames, blobs = self.sample_single(index)
        frames = self.transforms(frames)  # original size is (t, c, h, w)
        global_img_tensors = frames.permute(1, 0, 2, 3)  # (c, t, h, w)
        # 在这里分成两个 pathway
        if model_args.backbone == 'slowfast':
            fast_pathway = global_img_tensors
            slow_pathway = torch.index_select(global_img_tensors, 1, torch.linspace(0, global_img_tensors.shape[1]-1,
                                                                                    global_img_tensors.shape[1] // 8).long())
            global_img_tensors = [slow_pathway, fast_pathway]
        label = self.classes_dict[self.labels[index]]
        return global_img_tensors, blobs, label

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
