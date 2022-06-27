# 实现不同的 head, 包括
# global avg, region avg, global trajectory (attention), region trajectory (attention)
# 假定输入都是 3D feature map

import torch
import os
from torch.backends.cudnn import set_flags
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from compaction.model import composition_helpyer, mixup_helper
from compaction.model.common import Mlp
from compaction.model.mixup_helper import RandomMix, mixup_process
from compaction.model.reason_helper import BoxReasonNet, CategoryBoxEmbeddings, HOReasonNet, HOReasonSimp, VisReasonNet
from timm.models.layers import trunc_normal_
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

from . import vit_helper
from .detection_helper import box_to_normalized, build_region_feas, slice_bbox
# import vit_helper

_SF_STAGE_DIMS = [80, 320, 640, 1280, 2048]

class ReasonRegionHead(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.vis_net = VisReasonNet(cfg)
        self.box_net = BoxReasonNet(cfg)

        self.embed_dim = cfg.VIT.EMBED_DIM
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.head_dropout = 0.5
        self.cfg = cfg

        self.fuse_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.ReLU(inplace=True)
        )

        self.head_drop = nn.Dropout(p=self.head_dropout)
        self.fuse_classifier = nn.Linear(self.embed_dim, self.num_classes, bias=False)

    def forward(self, x, meta, labels=None):
        box_cls, box_reason_outs = self.box_net(x, meta, labels, use_feature=True)
        if self.training and self.cfg.TRAIN.MIXUP:
            vis_cls, vis_reason_outs, comp_cls, mix_cls, mix_label = self.vis_net(x, meta, labels, use_feature=True)
        else:
            vis_cls, vis_reason_outs = self.vis_net(x, meta, labels, use_feature=True)
        fuse_fea = torch.cat([vis_reason_outs.clone().detach(), box_reason_outs.clone().detach()], dim=-1)
        fuse_out = self.fuse_layer(fuse_fea)
        fuse_cls = self.fuse_classifier(self.head_drop(fuse_out))
        if self.training and self.cfg.TRAIN.MIXUP:
            return fuse_cls, vis_cls, box_cls, comp_cls, mix_cls, mix_label
        else:
            return fuse_cls, vis_cls, box_cls
        # return box_cls

class RegionVisReasonHeader(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.net = VisReasonNet(cfg)
        self.cfg = cfg

    def forward(self, x, meta, labels=None):
        if self.training and self.cfg.TRAIN.MIXUP:
            vis_cls, mix_cls, mix_label = self.net(x, meta, labels, use_feature=False)
            return vis_cls, mix_cls, mix_label
        else:
            vis_cls = self.net(x, meta, labels, use_feature=False)
            return vis_cls

class RegionComposerHeader(nn.Module):
    def __init__(self, cfg):
        super(RegionComposerHeader, self).__init__()
        dim_in = 2048
        self.crop_size = [3, 3]
        self.img_size = [224, 224]
        self.num_heads = cfg.VIT.NUM_HEADS
        self.nr_boxes = cfg.DATA.NUM_BOXES
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.head_dropout = cfg.VIT.HEAD_DROPOUT
        self.mlp_ratio = cfg.VIT.MLP_RATIO
        self.qkv_bias = cfg.VIT.QKV_BIAS
        self.drop_rate = cfg.VIT.DROP
        self.attn_drop_rate = cfg.VIT.ATTN_DROPOUT
        self.temporal_stride = cfg.SLOWFAST.ALPHA
        self.use_identity = cfg.REASON.IDENTITY
        self.spe_type = cfg.REASON.SPE
        self.cfg = cfg

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.conv5 = nn.Conv3d(
            dim_in,
            self.cfg.VIT.EMBED_DIM,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )

        self.embed_dim = cfg.VIT.EMBED_DIM
        self.pos_embed = vit_helper.PositionEmbeddingSine(
                cfg.VIT.EMBED_DIM // 3, attach=cfg.VIT.ATTACH_POS
            )

        self.obj_vis_embed = nn.Sequential(
                nn.Linear(
                    self.cfg.VIT.EMBED_DIM * self.crop_size[0] * self.crop_size[1],
                    self.cfg.VIT.EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(0.5))

        # self.action_memory = torch.randn((cfg.MODEL.NUM_CLASSES, self.embed_dim))
        # self.action_has_memory =set()
        # with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'object_relevant', 'val_oa_20_freq.npy'), 'rb') as f:
            # self.oa_prob = torch.from_numpy(np.load(f)).float()

        self.ho_reason = HOReasonSimp(dim = self.embed_dim,
                                      num_heads=self.num_heads,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=self.qkv_bias,
                                      drop=self.drop_rate,
                                      attn_drop=self.attn_drop_rate,
                                      norm_layer=norm_layer,
                                      depths=cfg.REASON.VIS_DEPTHS,
                                      use_cls=cfg.MODEL.USE_CLS,
                                      pos_each_layer=cfg.VIT.POS_EACH_LAYER,
                                      reason_types=['oh', 'ho', 'hh', 'oo'])

        # self.mixer = mixup_helper.RandomMix(cfg, k=3)
        self.mixer = mixup_helper.PriorMix(cfg, k=3)

        self.composition_classifier = nn.Linear(self.embed_dim, self.num_classes)

        self.head_drop = nn.Dropout(p=self.head_dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim, bias=False),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x, meta, labels=None):
        if isinstance(x, list):
            x = x[0]
        B, C, T, H, W = x.shape
        box_input = meta['object_boxes']
        roi_box_input = box_input[:, ::self.temporal_stride].contiguous() # b, T, nr_box, 4
        x = self.conv5(x) # b,c,t,h,w
        pos = self.pos_embed(x) # b,c,t,h,w
        x = torch.cat([x, pos], dim=1)

        x = x.permute(0, 2, 3, 4, 1).contiguous() # b, t, h, w, c
        x = x.reshape(B*T, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        roi_box_input = roi_box_input.view(B*T, self.nr_boxes, 4)
        boxes_list = box_to_normalized(roi_box_input, crop_size=[224, 224])
        region_vis_feas = build_region_feas(x, boxes_list,
                                            self.crop_size, self.img_size, flatten=False) # b*T*nr_box, d, 3, 3
        region_vis_feas = region_vis_feas.permute(0, 2, 3, 1).contiguous() # b*t*nr_box, 3, 3, d

        region_vis_feas, roi_pos = region_vis_feas.chunk(2, dim=-1) # b*t*nr_box, 3, 3, d

        # interaction
        vis_tokens = region_vis_feas.reshape(B, T, self.nr_boxes, self.crop_size[0]*self.crop_size[1], -1)
        roi_pos = roi_pos.reshape(B, T, self.nr_boxes, self.crop_size[0]*self.crop_size[1], -1)
        hand_vis_tokens = vis_tokens[:, :, :2] # b, t, 2, 3, 3, d
        obj_vis_tokens = vis_tokens[:, :, 2:] # b, t, n, 3, 3, d
        hand_pos = roi_pos[:, :, :2]
        obj_pos = roi_pos[:, :, 2:]
        hand_vis_tokens = hand_vis_tokens.reshape(B, -1, self.embed_dim)
        obj_vis_tokens = obj_vis_tokens.reshape(B, -1, self.embed_dim)
        hand_pos = hand_pos.reshape(B, -1, self.embed_dim)
        obj_pos = obj_pos.reshape(B, -1, self.embed_dim)
        ho = self.ho_reason(hand_vis_tokens, obj_vis_tokens, hand_pos, obj_pos) # b, N, d
        ho = torch.mean(ho, dim=1)

        cls_output = self.classifier(self.head_drop(ho))

        if self.training and self.cfg.TRAIN.MIXUP:
            # composition branch
            obj_vis_feas = region_vis_feas.reshape(B, T, self.nr_boxes, -1) # b,t,n,d
            obj_vis_feas = self.obj_vis_embed(obj_vis_feas)
            obj_vis_feas = torch.mean(obj_vis_feas, dim=1)[:, 2:] # b, n, d
            obj_indicator = meta['obj_indicator'][:, 2:]
            # mix_fea, mix_label = mixup_process(obj_vis_feas, labels)
            ori_comp, mix_fea, mix_label = self.mixer(obj_vis_feas, obj_indicator, labels)
            # with torch.no_grad():
            #     label_onehot = F.one_hot(labels.detach(), num_classes=self.num_classes).float()
            #     pred = self.composition_classifier(self.head_drop(mix_fea.detach()))
            #     pred = pred.view(B, -1, self.num_classes)
            #     y_hat = torch.matmul(pred, label_onehot.unsqueeze(-1)).squeeze()
            #     idx_best_comp = torch.argmax(y_hat, dim=1)
            # select_obj_fea = mix_fea[torch.arange(B), idx_best_comp]
            # mix_cls = self.composition_classifier(self.head_drop(select_obj_fea))
            # mix_label = mix_label[torch.arange(B), idx_best_comp]
            mix_cls = self.composition_classifier(self.head_drop(mix_fea.squeeze()))
            ori_cls = self.composition_classifier(self.head_drop(ori_comp.squeeze()))
            return cls_output, ori_cls, mix_cls, mix_label.squeeze()
        else:
            obj_vis_feas = region_vis_feas.reshape(B, T, self.nr_boxes, -1) # b,t,n,d
            obj_vis_feas = self.obj_vis_embed(obj_vis_feas)
            obj_vis_feas = torch.mean(obj_vis_feas, dim=1)[:, 2:] # b, n, d
            obj_vis_feas = torch.mean(obj_vis_feas, dim=1)
            comp_cls = self.composition_classifier(self.head_drop(obj_vis_feas))
            return cls_output, comp_cls

class RegionSemanticHeader(nn.Module):
    def __init__(self, cfg):
        super(RegionSemanticHeader, self).__init__()
        dim_in = 2048
        self.crop_size = [3, 3]
        self.img_size = [224, 224]
        self.num_heads = cfg.VIT.NUM_HEADS
        self.nr_boxes = cfg.DATA.NUM_BOXES
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.head_dropout = cfg.VIT.HEAD_DROPOUT
        self.mlp_ratio = cfg.VIT.MLP_RATIO
        self.qkv_bias = cfg.VIT.QKV_BIAS
        self.drop_rate = cfg.VIT.DROP
        self.attn_drop_rate = cfg.VIT.ATTN_DROPOUT
        self.temporal_stride = cfg.SLOWFAST.ALPHA
        self.use_identity = cfg.REASON.IDENTITY
        self.spe_type = cfg.REASON.SPE
        self.cfg = cfg

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.conv5 = nn.Conv3d(
            dim_in,
            self.cfg.VIT.EMBED_DIM,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )

        self.embed_dim = cfg.VIT.EMBED_DIM
        self.pos_embed = vit_helper.PositionEmbeddingSine(
                cfg.VIT.EMBED_DIM // 3, attach=cfg.VIT.ATTACH_POS
            )

        self.ho_reason = HOReasonSimp(dim = self.embed_dim,
                                      num_heads=self.num_heads,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=self.qkv_bias,
                                      drop=self.drop_rate,
                                      attn_drop=self.attn_drop_rate,
                                      norm_layer=norm_layer,
                                      depths=cfg.REASON.VIS_DEPTHS,
                                      use_cls=cfg.MODEL.USE_CLS,
                                      pos_each_layer=cfg.VIT.POS_EACH_LAYER,
                                      reason_types=['oh'])

        # self.composition_head = composition_helpyer.ComposerObjectHead(self.embed_dim, 'unseen', cfg)
        self.composition_head = composition_helpyer.ComposerBatch(self.embed_dim, 'unseen', cfg)

    def forward(self, x, meta, labels=None):
        if isinstance(x, list):
            x = x[0]
        B, C, T, H, W = x.shape
        box_input = meta['object_boxes']
        roi_box_input = box_input[:, ::self.temporal_stride].contiguous() # b, T, nr_box, 4
        x = self.conv5(x) # b,c,t,h,w
        pos = self.pos_embed(x) # b,c,t,h,w
        x = torch.cat([x, pos], dim=1)

        x = x.permute(0, 2, 3, 4, 1).contiguous() # b, t, h, w, c
        x = x.reshape(B*T, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        roi_box_input = roi_box_input.view(B*T, self.nr_boxes, 4)
        boxes_list = box_to_normalized(roi_box_input, crop_size=[224, 224])
        region_vis_feas = build_region_feas(x, boxes_list,
                                            self.crop_size, self.img_size, flatten=False) # b*T*nr_box, d, 3, 3
        region_vis_feas = region_vis_feas.permute(0, 2, 3, 1).contiguous() # b*t*nr_box, 3, 3, d

        region_vis_feas, roi_pos = region_vis_feas.chunk(2, dim=-1) # b*t*nr_box, 3, 3, d

        # action
        vis_tokens = region_vis_feas.reshape(B, T, self.nr_boxes, self.crop_size[0]*self.crop_size[1], -1)
        roi_pos = roi_pos.reshape(B, T, self.nr_boxes, self.crop_size[0]*self.crop_size[1], -1)
        hand_vis_tokens = vis_tokens[:, :, :2] # b, t, 2, 3, 3, d
        obj_vis_tokens = vis_tokens[:, :, 2:] # b, t, n, 3, 3, d
        hand_pos = roi_pos[:, :, :2]
        obj_pos = roi_pos[:, :, 2:]
        hand_vis_tokens = hand_vis_tokens.reshape(B, -1, self.embed_dim)
        obj_vis_tokens = obj_vis_tokens.reshape(B, -1, self.embed_dim)
        hand_pos = hand_pos.reshape(B, -1, self.embed_dim)
        obj_pos = obj_pos.reshape(B, -1, self.embed_dim)
        oh = self.ho_reason(hand_vis_tokens, obj_vis_tokens, hand_pos, obj_pos) # b, N, d
        oh = oh[0]
        # oh = oh.reshape(B, T, self.nr_boxes-2, self.crop_size[0]*self.crop_size[1], -1)
        # oh = torch.mean(oh, dim=1)
        # oh = torch.mean(oh, dim=2) # b, n, d
        oh = torch.mean(oh, dim=1)

        # if self.training:
        #     cls_output = self.composition_head(oh, meta, labels)
        # else:
        #     self.composition_head.prior_type = 'val'
        #     cls_output = self.composition_head(oh, meta, labels)
        cls_output = self.composition_head(oh, meta, labels)

        return cls_output

class BasicHead(nn.Module):
    def __init__(self, dim_in, dim_inner, num_classes, dropout_rate=0.0, act_func="softmax", inplace_relu=True) -> None:
        super(BasicHead, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
        self.inplace_relu = inplace_relu
        self._construct_head(dim_in, dim_inner)

    def _construct_head(self, dim_in, dim_inner, ):
        '''
        dim_in: 2048
        self.embed_dim: 512
        '''
        self.conv5 = nn.Conv2d(dim_in[0],
                               dim_inner,
                               kernel_size=(1, 1),
                               stride=1)
        self.avgpool3d = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)
        self.projection = nn.Linear(dim_inner+dim_in[1], self.num_classes, bias=True)

    def forward(self, x):
        '''
        x: (B, C, T, H, W)
        It often has the shape (8, 2048, 8, 14, 14)
        '''
        if isinstance(x, list):
            x, fast = x  # 这里和 slowfast 原本的做法有一些不同，我们没有融合两个分支，而是只取了 slow 分支，之后可能再实验先融合的做法
            fast = self.avgpool3d(fast).squeeze()
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)
        x = self.conv5(x)
        x = x.view(B, T, -1, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.avgpool3d(x).squeeze()

        if isinstance(fast, object):
            x = torch.cat([x, fast], dim=1)

        cls_output = self.projection(self.dropout(x))

        if not self.training:
            x = self.act(x)

        return cls_output

class RegionHead(nn.Module):
    def __init__(self, dim_in, dim_inner, num_classes, num_boxes, img_size, crop_size, temporal_stride, cfg=None) -> None:
        super(RegionHead, self).__init__()
        self.temporal_stride = temporal_stride
        self.nr_boxes = num_boxes
        self.img_size = img_size
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.cfg = cfg
        self._construct_head(dim_in, dim_inner)

    def _construct_head(self, dim_in, dim_inner):
        '''
        dim_in: 2048
        self.embed_dim: 512
        '''
        self.conv5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.region_embed = nn.Sequential(
            nn.Linear(
                dim_inner * self.crop_size[0] * self.crop_size[1],
                dim_inner), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.avgpool3d = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)
        # self.projection = nn.Linear(dim_inner, self.num_classes)
        # self.projection = nn.Sequential(
        #     nn.Linear(dim_inner, dim_inner),
        #     nn.Linear(dim_inner, self.num_classes)
        # )
        # self.mix_classifier = nn.Linear(dim_inner, self.num_classes)
        self.obj_classifier = nn.Sequential(
            nn.Linear(dim_inner, dim_inner),
            nn.Linear(dim_inner, 301)
        )
        self.projection = nn.Sequential(
            nn.Linear(dim_inner, dim_inner),
            nn.Linear(dim_inner, self.num_classes)
        )
        if self.cfg.TRAIN.MIXUP:
            self.mixer = RandomMix(self.cfg, k=1)

    def forward(self, x, meta, labels):
        if isinstance(x, list):
            x = x[0]
        B, C, T, H, W = x.shape
        x = self.conv5(x)

        box_input = meta['object_boxes']
        box_input = box_input[:, ::self.temporal_stride]
        box_input = box_input.contiguous()
        box_input = box_input.view(B*T, self.nr_boxes, 4)
        boxes_list = box_to_normalized(box_input, crop_size=[224, 224])

        x = x.permute(0, 2, 3, 4, 1).contiguous() # b, t, h, w, c
        x = x.reshape(B*T, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        region_vis_feas = build_region_feas(x, boxes_list,
                                            self.crop_size, self.img_size)
        region_vis_feas = self.region_embed(region_vis_feas)
        region_vis_feas = region_vis_feas.view(
            B, T, self.nr_boxes,
            region_vis_feas.size(-1)) # b, t, n, d

        region_vis_feas = torch.mean(region_vis_feas, dim=1) # b, n, d
        obj_vis_feas = region_vis_feas[:, 2:]
        obj_category = meta['obj_category'][:, 2:] # b, n

        obj_labels = []
        obj_feas = []
        for i in range(B):
            vid_obj_category = obj_category[i] # n
            cnt = 0
            for j in range(self.nr_boxes-2):
                # print(j)
                if vid_obj_category[j].item() != 0 and cnt < 2:
                    obj_labels.append(vid_obj_category[j])
                    obj_feas.append(obj_vis_feas[i][j])
                    cnt += 1
        obj_labels = torch.stack(obj_labels, dim=0)
        obj_feas = torch.stack(obj_feas, dim=0)
        obj_cls_output = self.obj_classifier(obj_feas)

        cls_output = self.projection(self.dropout(torch.mean(region_vis_feas, dim=1)))
        if self.training:
            return cls_output, obj_cls_output, obj_labels
        else:
            return cls_output

        
class HardDistangleHead(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.crop_size = [3, 3]
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.temporal_stride = cfg.SLOWFAST.ALPHA
        self.cfg = cfg
        self.embed_dim = cfg.VIT.EMBED_DIM
        self._construct_head()
        
    def _construct_head(self):
        dim_in = 2048
        dim_inner = self.embed_dim
        self.conv5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.region_embed = nn.Sequential(
            nn.Linear(dim_inner * self.crop_size[0] * self.crop_size[1],
                      dim_inner), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.dropout = nn.Dropout(0.5)
        self.projection = nn.Sequential(
            nn.Linear(dim_inner, dim_inner),
            nn.Linear(dim_inner, self.num_classes))
        
    def forward(self, x, meta, labels):
        if isinstance(x, list):
            x = x[0]
        B, C, T, H, W = x.shape
        x = self.conv5(x)

        # only use hand box
        box_input = meta['object_boxes']
        box_input = box_input[:, ::self.temporal_stride] # b, t, N, 4
        box_input = box_input.contiguous()
        hand_box, obj_box = box_input[:, :, :2], box_input[:, :, 2:]
        hand_box = hand_box.view(B*T, 2, 4)
        # perform roi over feature map
        boxes_list = box_to_normalized(hand_box, crop_size=[224, 224])
        region_vis_feas = build_region_feas(x, boxes_list, self.crop_size, self.img_size)
        region_vis_feas = self.region_embed(region_vis_feas)
        region_vis_feas = region_vis_feas.view(
            B, T, self.nr_boxes, region_vis_feas.size(-1)) # b, t, n, d
        region_vis_feas = torch.mean(region_vis_feas, dim=1) # b, n, d
        # action recognition
        cls_output = self.projection(self.dropout(torch.mean(region_vis_feas, dim=1)))
        return cls_output

if __name__ == '__main__':
    pass
