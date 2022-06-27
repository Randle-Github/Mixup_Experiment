import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.roi_align import roi_align
from model.slowfast.video_model_builder import SlowFast, cfg
from model.slowfast.load_checkpoint import load_checkpoint
import cfg_init

model_args = cfg_init.model_args()

def box_to_normalized(boxes_tensor, crop_size=[224, 224], mode='list'):
    # tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
    new_boxes_tensor = boxes_tensor.clone()
    new_boxes_tensor[..., 0] = (boxes_tensor[..., 0] -
                                boxes_tensor[..., 2] / 2.0) * crop_size[0]
    new_boxes_tensor[..., 1] = (boxes_tensor[..., 1] -
                                boxes_tensor[..., 3] / 2.0) * crop_size[1]
    new_boxes_tensor[..., 2] = (boxes_tensor[..., 0] +
                                boxes_tensor[..., 2] / 2.0) * crop_size[0]
    new_boxes_tensor[..., 3] = (boxes_tensor[..., 1] +
                                boxes_tensor[..., 3] / 2.0) * crop_size[1]
    if mode == 'list':
        boxes_list = []
        for boxes in new_boxes_tensor:
            boxes_list.append(boxes)
        return boxes_list
    elif mode == 'tensor':
        return new_boxes_tensor

def build_region_feas(feature_maps,
                      boxes_list,
                      output_crop_size=[3, 3],
                      img_size=[224, 224]):
    # Building feas for each bounding box by using RoI Align
    # feature_maps:[N,C,H,W], where N=b*T
    IH, IW = img_size
    FH, FW = feature_maps.size()[-2:]  # Feature_H, Feature_W
    region_feas = roi_align(feature_maps,
                            boxes_list,
                            output_crop_size,
                            spatial_scale=float(FW) /
                            IW)  # b*T*K, C, S, S; S denotes output_size
    return region_feas.view(region_feas.size(0), -1)  # b*T*K, D*S*S
        
class BboxVisualModel(nn.Module):
    '''
    backbone: i3d
    '''

    def __init__(self, opt):
        nn.Module.__init__(self)
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames
        self.nr_boxes = opt.num_boxes
        self.use_region = False  # 决定是否用 ROI feature
        
        self.img_feature_dim = 512

        self.backbone = SlowFast(cfg)
        load_checkpoint('./ckpt/SLOWFAST_8x8_R50.pkl', self.backbone, data_parallel=False, convert_from_caffe2=True)

        self.conv = nn.Conv2d(2048,
                              self.img_feature_dim,
                              kernel_size=(1, 1),
                              stride=1)
        self.crop_size = [3, 3]
        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.avgpool3d = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)

        self.region_vis_embed = nn.Sequential(
            nn.Linear(
                self.img_feature_dim * self.crop_size[0] * self.crop_size[1],
                self.img_feature_dim), nn.ReLU(inplace=True), nn.Dropout(0.5))

        self.global_embed = nn.Sequential(
                nn.Linear(
                    2048,
                    self.img_feature_dim), nn.ReLU(inplace=True), nn.Dropout(0.5))

        self.classifier = nn.Sequential(
            nn.Linear(self.img_feature_dim, self.img_feature_dim),
            nn.ReLU(inplace=True), nn.Linear(self.img_feature_dim, 512),
            nn.ReLU(inplace=True), nn.Linear(512, self.nr_actions))

        self.fc = nn.Linear(512, self.nr_actions)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(
            new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, blobs, extract_global=False, is_inference=False):
        '''
        b, _, T, H, W  = global_img_input.size()
        global_img_input = global_img_input.permute(0, 2, 1, 3, 4).contiguous()
        global_img_input = global_img_input.view(b*T, 3, H, W)
        org_feas = self.backbone(global_img_input) # (b*T, 2048)
        conv_fea_maps = self.conv(org_feas)  # (b*T, img_feature_dim)
        box_tensors = box_input.view(b * T, self.nr_boxes, 4)
        '''
        box_input = blobs['object_boxes']
        
        org_feas = self.backbone(global_img_input)
        if isinstance(org_feas, list):
            org_feas, fast_fea = org_feas

        b, _, T, H, W = org_feas.size()
        org_feas = org_feas.permute(0, 2, 1, 3, 4).contiguous()
        org_feas = org_feas.view(b * T, 2048, H, W)
        conv_fea_maps = self.conv(org_feas)  # (b*T, img_feature_dim, h, w)

        if self.use_region:
            if box_input.size(1) != T:
                # box_input = box_input[::2]  # FIXME: 这里不对，第 0 维是 batch. 之前 train 的 slowfast 就存在这个问题
                box_input = box_input[:, ::2, ...]
                box_input = box_input.contiguous()
            box_tensors = box_input.view(b * T, self.nr_boxes, 4)

            boxes_list = box_to_normalized(box_tensors, crop_size=[224, 224])
            if isinstance(global_img_input, list):
                img_size = global_img_input[0].size()[-2:]
            else:
                img_size = global_img_input.size()[-2:]

            # (b*T*nr_boxes, C), C=3*3*d
            region_vis_feas = build_region_feas(conv_fea_maps, boxes_list,
                                                self.crop_size, img_size)

            region_vis_feas = self.region_vis_embed(region_vis_feas)

            region_vis_feas = region_vis_feas.view(
                b, T, self.nr_boxes,
                region_vis_feas.size(-1))  # (b, t, n, dim)
            # region_vis_feas = region_vis_feas.transpose(2, 1).contiguous()  # (b, n, t, img_feature_dim)
            # region_vis_feas = region_vis_feas.permute(0, 3, 2, 1).contiguous()
            region_vis_feas = region_vis_feas.permute(
                0, 3, 1, 2).contiguous()  # (b, d, n, t)
            region_vis_feas = self.avgpool2d(region_vis_feas).squeeze()
            
        else: # do not use region feature
            region_vis_feas = conv_fea_maps.view(b, T, -1, H, W) # (b, t, dim)
            region_vis_feas = region_vis_feas.permute(0, 2, 1, 3, 4).contiguous() # (b, dim, T, H, W)
            # print(region_vis_feas.size())
            region_vis_feas = self.avgpool3d(region_vis_feas).squeeze()
            # print(region_vis_feas.size())
        cls_output = self.fc(self.dropout(region_vis_feas))

        return cls_output

        # region_vis_feas = torch.mean(region_vis_feas, dim=2)  # (b, n, dim)
        # global_features = self.avgpool(region_vis_feas).squeeze()
        # global_features = torch.mean(region_vis_feas, dim=1)  # (b, dim)
        # global_features = self.dropout(global_features)

        # cls_output = self.fc(global_features)
        # cls_output = self.classifier(global_features)
        # return cls_output

class BboxInteractionLatentModel(nn.Module):
    '''
    Add bbox category embedding
    '''

    def __init__(self, opt):
        nn.Module.__init__(self)
        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        # self.nr_frames = opt.num_frames
        self.nr_frames = opt.num_frames // 2
        self.coord_feature_dim = opt.coord_feature_dim

        self.interaction = nn.Sequential(
            nn.Linear(self.nr_boxes * 4,
                      self.coord_feature_dim // 2,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim // 2,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU())

        self.category_embed_layer = nn.Embedding(3,
                                                 opt.coord_feature_dim // 2,
                                                 padding_idx=0,
                                                 scale_grad_by_freq=True)

        # Fusion of Object Interaction and Category Embedding
        self.fuse_layer = nn.Sequential(
            nn.Linear(self.coord_feature_dim + self.coord_feature_dim // 2,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True))

        self.temporal_aggregate_func = nn.Sequential(
            nn.Linear(self.nr_frames * self.coord_feature_dim,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU())

        self.object_compose_func = nn.Sequential(
            nn.Linear(self.nr_boxes * self.coord_feature_dim,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True), nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True), nn.Linear(512, self.nr_actions))

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(
            new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, box_categories, box_input, extract_fea=False):
        b, _, _, _ = box_input.size()  # (b, T, nr_boxes, 4)

        box_categories = box_categories.long()

        box_categories = box_categories.view(b * self.nr_frames * self.nr_boxes)
        box_category_embeddings = self.category_embed_layer(box_categories)
        identity_repre = box_category_embeddings.view(
            b, self.nr_frames, self.nr_boxes,
            -1)  # (b, t, n, coord_feature_dim//2)

        # Calculate the distance vector between objects
        box_dis_vec = box_input.unsqueeze(3) - box_input.unsqueeze(
            2)  # (b, T, nr_boxes, nr_boxes, 4)

        box_dis_vec_inp = box_dis_vec.view(b * self.nr_frames * self.nr_boxes,
                                           -1)
        inter_fe = self.interaction(box_dis_vec_inp)
        inter_fe = inter_fe.view(b, self.nr_frames, self.nr_boxes,
                                 -1)  # (b, T, nr_boxes, coord_feature_dim)

        inter_fea_latent = torch.cat([inter_fe, identity_repre],
                                     dim=-1)  # (b, T, nr_boxes, dim+dim//2)
        inter_fea_latent = inter_fea_latent.view(-1,
                                                 inter_fea_latent.size()[-1])

        inter_fe = self.fuse_layer(inter_fea_latent)
        inter_fe = inter_fe.view(
            b, self.nr_frames, self.nr_boxes,
            -1).transpose(2,
                          1).contiguous()  # (b, nr_boxes, T, coord_feture_dim)

        inter_fea_inp = inter_fe.view(b * self.nr_boxes, -1)
        frame_inter_fea = self.temporal_aggregate_func(inter_fea_inp)
        frame_inter_fea = frame_inter_fea.view(
            b, self.nr_boxes, -1)  # (b, nr_boxes, coord_feature_dim)

        obj_fe = frame_inter_fea

        obj_inter_fea_inp = frame_inter_fea.view(b, -1)
        video_fe = self.object_compose_func(obj_inter_fea_inp)

        if extract_fea:
            return video_fe
        
        cls_output = self.classifier(video_fe)

        return cls_output, video_fe

# TODO: 现在的 bboxinteraction 是先 temporal 聚合再 object 聚合，如果换一下方向会怎么样？
class BboxInteractionLatentModel2(nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames // 2
        self.coord_feature_dim = opt.coord_feature_dim

        self.interaction = nn.Sequential(
            nn.Linear(self.nr_boxes * 4,
                      self.coord_feature_dim // 2,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim // 2,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU())

        self.category_embed_layer = nn.Embedding(3,
                                                 opt.coord_feature_dim // 2,
                                                 padding_idx=0,
                                                 scale_grad_by_freq=True)

        # Fusion of Object Interaction and Category Embedding
        self.fuse_layer = nn.Sequential(
            nn.Linear(self.coord_feature_dim + self.coord_feature_dim // 2,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True))

        self.frame_aggregate_func = nn.Sequential(
            nn.Linear(self.nr_boxes * self.coord_feature_dim,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU())

        self.temporal_func = nn.Sequential(
            nn.Linear(self.nr_frames * self.coord_feature_dim,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim,
                      bias=False), nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True), nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True), nn.Linear(512, self.nr_actions))

    def forward(self, box_categories, box_input, is_inference=False):
        b, _, _, _ = box_input.size()  # (b, T, nr_boxes, 4)

        box_categories = box_categories.long()

        box_categories = box_categories.view(b * self.nr_frames * self.nr_boxes)
        box_category_embeddings = self.category_embed_layer(box_categories)
        identity_repre = box_category_embeddings.view(
            b, self.nr_frames, self.nr_boxes,
            -1)  # (b, t, n, coord_feature_dim//2)

        # Calculate the distance vector between objects
        box_dis_vec = box_input.unsqueeze(3) - box_input.unsqueeze(
            2)  # (b, T, nr_boxes, nr_boxes, 4)

        box_dis_vec_inp = box_dis_vec.view(b * self.nr_frames * self.nr_boxes,
                                           -1)
        inter_fe = self.interaction(box_dis_vec_inp)
        inter_fe = inter_fe.view(b, self.nr_frames, self.nr_boxes,
                                 -1)  # (b, T, nr_boxes, coord_feature_dim)

        inter_fea_latent = torch.cat([inter_fe, identity_repre],
                                     dim=-1)  # (b, T, nr_boxes, dim+dim//2)
        inter_fea_latent = inter_fea_latent.view(-1,
                                                 inter_fea_latent.size()[-1])

        inter_fe = self.fuse_layer(inter_fea_latent)
        inter_fe = inter_fe.view(
            b, self.nr_frames, self.nr_boxes, -1)  # (b, T, n, coord_feture_dim)

        inter_fea_inp = inter_fe.view(b * self.nr_frames, -1)
        frame_inter_fea = self.frame_aggregate_func(inter_fea_inp)
        frame_inter_fea = frame_inter_fea.view(
            b, self.nr_frames, -1)  # (b, T, coord_feature_dim)

        frame_inter_fea_inp = frame_inter_fea.view(b, -1)
        video_fe = self.temporal_func(frame_inter_fea_inp)

        cls_output = self.classifier(video_fe)

        return cls_output, video_fe
