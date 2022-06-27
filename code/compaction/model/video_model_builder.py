import torch
import torch.nn as nn
import torch.nn.functional as F
from compaction.model import common, composition_helpyer, reason_helper, trunk_helper, vit_helper
from compaction.model.reason_helper import BoxReasonNet, HOReasonSimp
from compaction.model.batchnorm_helper import get_norm
from compaction.model.detection_helper import box_to_normalized, build_region_feas

import compaction.model.weight_init_helper as init_helper
import compaction.model.stem_helper as stem_helper
import compaction.model.resnet_helper as resnet_helper
import compaction.model.head_helper as head_helper
import compaction.utils.logging as logging
from functools import partial
import numpy as np

from .build import MODEL_REGISTRY

logger = logging.get_logger(__name__)
# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}

class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]

@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        # self.norm_module = nn.BatchNorm3d
        # self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self.cfg = cfg
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        # assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1['slowfast']
        assert len({len(pool_size), self.num_pathways}) == 1
        # assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()  # resnet50 or resnet101

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        width_per_group = 64
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS['slowfast']

        # stem 是指对 input 数据处理的第一层
        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )

        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.MODEL.HEAD == 'basic':
            self.head = head_helper.BasicHead(
                dim_in=(2048, 256),
                dim_inner=512,
                num_classes=cfg.MODEL.NUM_CLASSES,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )
        elif cfg.MODEL.HEAD == 'region':
            self.head = head_helper.RegionHead(
                dim_in=2048,
                dim_inner=cfg.VIT.EMBED_DIM,
                num_classes=cfg.MODEL.NUM_CLASSES,
                num_boxes=cfg.DATA.NUM_BOXES,
                img_size=(224, 224),
                crop_size=(3, 3),
                temporal_stride=cfg.SLOWFAST.ALPHA,  # 时间维度上的降采样倍数
                cfg=cfg
            )
        elif cfg.MODEL.HEAD == 'hard_distangle':
            self.head = head_helper.HardDistangleHead(
                cfg
            )
        elif cfg.MODEL.HEAD == 'reason_region':
            self.head = head_helper.ReasonRegionHead(
                cfg
            )
        elif cfg.MODEL.HEAD == 'reason_region_vis':
            self.head = head_helper.RegionVisReasonHeader(
                cfg
            )
        elif cfg.MODEL.HEAD == 'region_composer':
            self.head = head_helper.RegionComposerHeader(
                cfg
            )
        elif cfg.MODEL.HEAD == 'region_semantic':
            self.head = head_helper.RegionSemanticHeader(
                cfg
            )
        else:
            raise NotImplementedError(f"The head type: {cfg.MODEL.HEAD} not implement yet!")

        if cfg.TRAIN.FS_FINETUNE:
            self.finetune()

    def forward(self, x, meta=None, labels=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        # TODO: s5 这里还需要再 fuse fast
        x = self.s5(x)
        x = self.head(x, meta, labels)
        return x

    def finetune(self):
        frozen_weights = 0
        for name, param in self.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
                logger.info('Training : {}'.format(name))
            else:
                param.requires_grad = False
                frozen_weights += 1
        logger.info('Number of frozen weights {}'.format(frozen_weights))

@MODEL_REGISTRY.register()
class STLT(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.net = head_helper.BoxReasonNet(cfg)
        self.coord_feature_dim = 768
        self.nr_actions = 174
        self.visual_embedder = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.ReLU(inplace=True),
        )

        self.verb_embedder = nn.Embedding(self.nr_actions, 300)
        pretrained_weight = self.load_word_embeddings('verb')
        self.verb_embedder.weight.data.copy_(pretrained_weight)
        self.projection = nn.Linear(300, self.coord_feature_dim)
        self.scale = 20

    def load_word_embeddings(self, mode='verb'):
        import os
        import numpy as np
        HOME_DIR = os.path.expanduser('~')
        emb = np.load(os.path.join(HOME_DIR, 'FSL_Datasets', 'sth-sth-v2', 'object_relevant', f'{mode}_embedding.npy'))
        return torch.from_numpy(emb)

    def forward(self, x, meta, labels=None):
        box_cls, vid_feats = self.net(x, meta, labels, use_feature=True)
        vid_feats = self.visual_embedder(vid_feats) # b, 300
        verb_feats = self.projection(self.verb_embedder(torch.arange(self.nr_actions, device=vid_feats.device))) # 174, 300
        vid_feats = F.normalize(vid_feats, dim=1)
        verb_feats = F.normalize(verb_feats, dim=1).permute(1, 0).contiguous() # 300, 174

        cls_output = self.scale * torch.matmul(vid_feats, verb_feats) # b, 174
        box_cls = cls_output

        return box_cls

@MODEL_REGISTRY.register()
class STLTPlus(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.embed_dim = cfg.VIT.EMBED_DIM
        self.num_heads = cfg.VIT.NUM_HEADS
        self.nr_boxes = cfg.DATA.NUM_BOXES
        self.nr_frame = cfg.DATA.NUM_FRAMES
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.head_dropout = cfg.VIT.HEAD_DROPOUT
        self.mlp_ratio = cfg.VIT.MLP_RATIO
        self.qkv_bias = cfg.VIT.QKV_BIAS
        self.drop_rate = cfg.VIT.DROP
        self.attn_drop_rate = cfg.VIT.ATTN_DROPOUT
        self.temporal_stride = cfg.SLOWFAST.ALPHA
        self.use_identity = cfg.REASON.IDENTITY
        self.box_pretrain = cfg.REASON.BOX_PRETRAIN
        self.spe_type = cfg.REASON.SPE
        self.cfg = cfg
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_encoder_layer = 2
        self.encoder = nn.ModuleList([vit_helper.SelfAttentionBlock(
            dim = self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            norm_layer=norm_layer,
        ) for _ in range(self.num_encoder_layer)])

        self.box_reason = reason_helper.HOReasonNet(
            dim = self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            norm_layer=norm_layer,
            depths = cfg.REASON.BOX_DEPTHS,
            use_cls=cfg.MODEL.USE_CLS,
            pos_each_layer=cfg.VIT.POS_EACH_LAYER
        )
        self.box_embed = reason_helper.CategoryBoxEmbeddings(self.embed_dim)
        self.box_temp_embed = nn.Embedding(
            cfg.DATA.NUM_FRAMES, self.embed_dim
        )
        self.register_buffer(
            "position_ids", torch.arange(cfg.DATA.NUM_FRAMES).expand((1, -1))
        )

        self.box_classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_classes)
        )
        self.head_drop = nn.Dropout(p=self.head_dropout)

        self.identity_embedding = nn.Embedding(3,
                                               self.embed_dim,
                                               padding_idx=0,
                                               scale_grad_by_freq=True)

        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.embed_dim*2, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU()
        )

    def forward(self, x, meta, labels=None):
        box_input = meta['object_boxes']
        box_category = meta['box_category'] # b, t, N

        B = box_input.size(0)
        box_category = box_category.long()
        box_category = box_category.flatten()
        box_category_embeddings = self.identity_embedding(box_category)
        identity_repre = box_category_embeddings.view(
            B, -1, self.nr_boxes, self.embed_dim) # B, T(32), n, d

        box_tokens = self.box_embed(box_input) # b, T, n, d

        # 用同 frame 的其他 box 来增强表示当前结点
        bf = box_tokens.transpose(1, 2).contiguous() # b, n, t, d
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, t, d)
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself
        bf_and_message = torch.cat([bf, spatial_message], dim=3)  # (b, n, t, 2*d)
        bf_spatial = self.spatial_node_fusion(bf_and_message.view(B*self.nr_boxes*self.nr_frame, -1))
        bf_spatial = bf_spatial.view(B, self.nr_boxes, self.nr_frame, -1)
        box_tokens = bf_spatial.transpose(1, 2).contiguous() # b, t, n, d

        box_temp_pe = self.box_temp_embed(self.position_ids[:, :box_input.size(1)]) # b,t,d
        box_temp_pe = box_temp_pe.unsqueeze(-2).expand(-1, -1, self.nr_boxes, -1)
        hand_pe  = box_temp_pe[:, :, :2] + identity_repre[:, :, :2]
        obj_pe  = box_temp_pe[:, :, 2:] + identity_repre[:, :, 2:]
        box_tokens = box_tokens + box_temp_pe + identity_repre # b, t, n, d

        # 先对 box_tokens 进行 self-attention, 增强表达
        # box_tokens = box_tokens.reshape(B, -1, self.cfg.VIT.EMBED_DIM)
        # for i, blk in enumerate(self.encoder):
            # box_tokens = blk(box_tokens)

        box_tokens = box_tokens.reshape(B, self.nr_frame, self.nr_boxes, -1)
        hand_box_tokens = box_tokens[:, :, :2]
        obj_box_tokens = box_tokens[:, :, 2:]

        hand_box_tokens = hand_box_tokens.reshape(B, -1, self.cfg.VIT.EMBED_DIM)
        obj_box_tokens = obj_box_tokens.reshape(B, -1, self.cfg.VIT.EMBED_DIM)
        hand_pe = hand_pe.reshape(B, -1, self.cfg.VIT.EMBED_DIM)
        obj_pe = obj_pe.reshape(B, -1, self.cfg.VIT.EMBED_DIM)

        box_reason_outs = self.box_reason(hand_box_tokens, obj_box_tokens, hand_pe, obj_pe)
        # print(box_reason_outs[0].size())
        box_reason_outs = torch.cat(box_reason_outs, dim=1)
        box_reason_outs = torch.mean(box_reason_outs, dim=1)

        box_cls = self.box_classifier(self.head_drop(box_reason_outs))
        return box_cls

@MODEL_REGISTRY.register()
class STIN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.nr_boxes = cfg.DATA.NUM_BOXES
        self.nr_actions = cfg.MODEL.NUM_CLASSES
        self.nr_frames = cfg.DATA.NUM_FRAMES  # 原来是 num_frames//2
        self.coord_feature_dim = cfg.VIT.EMBED_DIM
        self.cfg = cfg

        self.category_embed_layer = nn.Embedding(3, cfg.VIT.EMBED_DIM // 2, padding_idx=0, scale_grad_by_freq=True)

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512), #self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        self.use_composition = True
        # self.composition_head = composition_helpyer.ComposerHead(self.coord_feature_dim, 'unseen', cfg)
        self.composition_head = composition_helpyer.ComposerBatch(self.coord_feature_dim, 'unseen', cfg)

    def forward(self, x, meta, labels=None):
        box_input = meta['object_boxes']
        box_categories = meta['box_category'] # b, t, N
        b = box_input.size(0)
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b*self.nr_boxes*self.nr_frames, 4)

        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(b*self.nr_boxes*self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories)  # (b*nr_b*nr_f, coord_feature_dim//2)

        bf = self.coord_to_feature(box_input)
        bf = torch.cat([bf, box_category_embeddings], dim=1)  # (b*nr_b*nr_f, coord_feature_dim + coord_feature_dim//2)
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        bf = bf.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself
        bf_and_message = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.spatial_node_fusion(bf_and_message.view(b*self.nr_boxes*self.nr_frames, -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, self.nr_frames*self.coord_feature_dim)

        box_features = self.box_feature_fusion(bf_temporal_input.view(b*self.nr_boxes, -1))  # (b*nr_boxes, coord_feature_dim)
        box_features = box_features.view(b, self.nr_boxes, -1)
        box_features = torch.mean(box_features, dim=1)  # (b, coord_feature_dim)
        video_features = box_features

        # composition
        if self.use_composition:
            cls_output = self.composition_head(video_features, meta, labels)
        else:
            cls_output = self.classifier(video_features)

        return cls_output

