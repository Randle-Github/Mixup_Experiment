from typing import OrderedDict
from torch._C import _tracer_warn_use_python
import torch.nn as nn
from torch.nn.functional import embedding
from torch.nn.init import trunc_normal_
import torch
import numpy as np
from compaction.datasets.utils import spatial_sampling
import torch.nn.functional as F
from functools import partial
from compaction.model import mixup_helper
import compaction.utils.logging as logging

from compaction.model.vit_helper import HOReasonBlock, SelfAttentionBlock
from .detection_helper import box_to_normalized, build_region_feas
from . import vit_helper
from compaction.model.mixup_helper import mixup_process, token_mix

logger = logging.get_logger(__name__)

class CategoryBoxEmbeddings(nn.Module):
    def __init__(self, hidden_size, ):
        super(CategoryBoxEmbeddings, self).__init__()
        self.box_embedding = nn.Linear(4, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, box_input) -> torch.Tensor:
        boxes_embeddings = self.box_embedding(box_input)
        embeddings = boxes_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class HOReasonNet(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, norm_layer, depths, use_cls=True, attn_type='traj', pos_each_layer=False) -> None:
        super().__init__()
        self.reason_types = ['ho', 'hh', 'oh', 'oo']
        assert isinstance(depths, list) and len(depths)==2
        self.embed_dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop
        self.attn_drop_rate = attn_drop
        self.use_cls = use_cls
        self.num_ho_layers, self.num_aggregate_layers = depths
        self.pos_each_layer = pos_each_layer

        for r in self.reason_types:
            blk = nn.ModuleList([
                HOReasonBlock(
                    dim=self.embed_dim,
                    num_heads =self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    norm_layer=norm_layer,
                    attn_type=attn_type
                ) for _ in range(self.num_ho_layers)
            ])
            setattr(self, r, blk) # eg. self.ho = blk

            # add cls token for each reason type
            cls = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(cls, std=.02)
            setattr(self, r+'cls', cls)  # eg. self.ho_cls = cls

            if self.num_aggregate_layers > 0:
                blk = nn.ModuleList([
                    SelfAttentionBlock(
                        dim=self.embed_dim,
                        num_heads =self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=self.qkv_bias,
                        drop=self.drop_rate,
                        attn_drop=self.attn_drop_rate,
                        norm_layer=norm_layer
                    ) for _ in range(self.num_aggregate_layers)
                ])
            else:
                blk = nn.ModuleList([nn.Identity()])

            setattr(self, r+'agg', blk) # eg self.ho_agg = blk

    def with_pos_embed(self, src, pos):
        if self.pos_each_layer:
            return src+pos
        else:
            return src

    def forward(self, h, o, hpos, opos):
        # h, o 是从 feature map 处理得到的 tokens
        # hpos, opos 是 token 对应的 position embedding
        # h,o : (b, n, d); hpos, opos: (b, n, d)

        # 最后返回四种类型的交互表示
        B, _, _ = h.shape
        # cross attention
        blks = getattr(self, 'ho')
        for i, blk in enumerate(blks):
            if i == 0:
                h = self.with_pos_embed(h, hpos)
                o = self.with_pos_embed(o, opos)
                ho = blk(h, o)
            else:
                ho = self.with_pos_embed(ho, hpos)
                ho = blk(ho, o)
        blks = getattr(self, 'hh')
        for i, blk in enumerate(blks):
            if i == 0:
                h = self.with_pos_embed(h, hpos)
                o = self.with_pos_embed(o, opos)
                hh = blk(h, h)
            else:
                hh = self.with_pos_embed(hh, hpos)
                hh = blk(hh, h)

        blks = getattr(self, 'oh')
        for i, blk in enumerate(blks):
            if i == 0:
                o = self.with_pos_embed(o, opos)
                h = self.with_pos_embed(h, hpos)
                oh = blk(o, h)
            else:
                oh = self.with_pos_embed(oh, opos)
                h = self.with_pos_embed(h, hpos)
                oh = blk(oh, h)

        blks = getattr(self, 'oo')
        for i, blk in enumerate(blks):
            if i == 0:
                o = self.with_pos_embed(o, opos)
                oo = blk(o, o)
            else:
                oo = self.with_pos_embed(oo, opos)
                oo = blk(oo, o)

        # cat cls token for each type
        outs = [ho, hh, oh, oo]
        if self.use_cls:
            for i, r in enumerate(self.reason_types):
                cls_token = getattr(self, r+'cls')
                cls_token = cls_token.expand(B, -1, -1)
                outs[i] = torch.cat((cls_token, outs[i]), dim=1)

        # self attention for aggregation
        for i, r in enumerate(self.reason_types):
            blks = getattr(self, r+'agg')
            x = outs[i]
            for blk in blks:
                x = blk(x)
            outs[i] = x

        if self.use_cls:
            return [out[:, 0].unsqueeze(1) for out in outs] # [B, 1, d]
        else:
            return outs # [B, N, d]

class HOReasonSimp(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, norm_layer, depths, use_cls=True, attn_type='traj', pos_each_layer=False, reason_types=['ho']) -> None:
        super().__init__()
        self.reason_types = reason_types
        assert isinstance(depths, list) and len(depths)==2
        self.embed_dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop
        self.attn_drop_rate = attn_drop
        self.use_cls = use_cls
        self.num_ho_layers, self.num_aggregate_layers = depths
        self.pos_each_layer = pos_each_layer

        for r in self.reason_types:
            blk = nn.ModuleList([
                HOReasonBlock(
                    dim=self.embed_dim,
                    num_heads =self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    norm_layer=norm_layer,
                    attn_type=attn_type
                ) for _ in range(self.num_ho_layers)
            ])
            setattr(self, r, blk) # eg. self.ho = blk

            if self.num_aggregate_layers > 0:
                blk = nn.ModuleList([
                    SelfAttentionBlock(
                        dim=self.embed_dim,
                        num_heads =self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=self.qkv_bias,
                        drop=self.drop_rate,
                        attn_drop=self.attn_drop_rate,
                        norm_layer=norm_layer
                    ) for _ in range(self.num_aggregate_layers)
                ])
            else:
                blk = nn.ModuleList([nn.Identity()])

            setattr(self, r+'agg', blk) # eg self.ho_agg = blk

    def with_pos_embed(self, src, pos):
        if self.pos_each_layer:
            return src+pos
        else:
            return src

    def forward(self, h, o, hpos, opos):
        B, _, _ = h.shape

        outs = []
        for r in self.reason_types:
            blks = getattr(self, r)
            for i, blk in enumerate(blks):
                if i == 0:
                    h = self.with_pos_embed(h, hpos)
                    o = self.with_pos_embed(o, opos)
                    if r in ['ho', 'hh']:
                        out = blk(h, o)
                    else:
                        out = blk(o, h)
                else:
                    if r in ['ho', 'hh']: # hand as query
                        out = self.with_pos_embed(out, hpos)
                        out = blk(out, o)
                    else: # obj as query
                        out = self.with_pos_embed(out, opos)
                        out = blk(out, h)
            outs.append((out, r))
        
        # self attention for aggregation
        for i, (out, r) in enumerate(outs):
            blks = getattr(self, f'{r}agg')
            for blk in blks:
                out = blk(out)
            outs[i] = out
            
        # outs = torch.cat(outs, dim=-1)
        return outs # [B, N, d]

class BoxReasonNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.embed_dim = cfg.VIT.EMBED_DIM
        self.num_heads = cfg.VIT.NUM_HEADS
        self.nr_boxes = cfg.DATA.NUM_BOXES
        self.num_classes = cfg.MODEL.NUM_CLASSES
        # self.head_dropout = cfg.VIT.HEAD_DROPOUT
        self.head_dropout = 0.2
        self.mlp_ratio = cfg.VIT.MLP_RATIO
        self.qkv_bias = cfg.VIT.QKV_BIAS
        # self.drop_rate = cfg.VIT.DROP
        self.drop_rate = 0.2
        self.attn_drop_rate = cfg.VIT.ATTN_DROPOUT
        self.box_pretrain = cfg.REASON.BOX_PRETRAIN
        self.cfg = cfg
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.box_reason = HOReasonNet(
            dim = self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            norm_layer=norm_layer,
            depths = cfg.REASON.BOX_DEPTHS,
            use_cls=cfg.REASON.BOX_USE_CLS,
            pos_each_layer=False
        )
        self.box_embed = CategoryBoxEmbeddings(self.embed_dim)
        self.box_temp_embed = nn.Embedding(
            cfg.DATA.NUM_FRAMES, self.embed_dim
        )
        self.register_buffer(
            "position_ids", torch.arange(cfg.DATA.NUM_FRAMES).expand((1, -1))
        )

        self.box_lin = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.box_classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.head_drop = nn.Dropout(p=self.head_dropout)

        self.identity_embedding = nn.Embedding(3,
                                               self.embed_dim,
                                               padding_idx=0,
                                               scale_grad_by_freq=True)

        if self.box_pretrain:
            self.load_pretrain(self.box_pretrain)

    def forward(self, x, meta, labels=None, use_feature=False):
        box_input = meta['object_boxes'].clone().detach()
        box_category = meta['box_category'].clone().detach()

        B = box_input.size(0)
        box_category = box_category.long()
        box_category = box_category.flatten()
        box_category_embeddings = self.identity_embedding(box_category)
        identity_repre = box_category_embeddings.view(
            B, -1, self.nr_boxes, self.embed_dim) # B, T(32), n, d

        box_tokens = self.box_embed(box_input) # b, T, n, d
        box_temp_pe = self.box_temp_embed(self.position_ids[:, :box_input.size(1)]) # b,t,d
        box_temp_pe = box_temp_pe.unsqueeze(-2).expand(-1, -1, self.nr_boxes, -1)
        box_tokens = box_tokens + box_temp_pe + identity_repre
        hand_box_tokens = box_tokens[:, :, :2]
        obj_box_tokens = box_tokens[:, :, 2:]
        hand_box_tokens = hand_box_tokens.reshape(B, -1, self.cfg.VIT.EMBED_DIM)
        obj_box_tokens = obj_box_tokens.reshape(B, -1, self.cfg.VIT.EMBED_DIM)

        box_reason_outs = self.box_reason(hand_box_tokens, obj_box_tokens, None, None)
        box_reason_outs = torch.cat(box_reason_outs, dim=1)
        box_reason_outs = torch.mean(box_reason_outs, dim=1)

        box_reason_outs = self.box_lin(box_reason_outs)
        box_cls = self.box_classifier(self.head_drop(box_reason_outs))
        if use_feature:
            return box_cls, box_reason_outs
        else:
            return box_cls

    def load_pretrain(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        # logger.info('box param names: ')
        # for name, param in self.named_parameters():
            # logger.info(name)

        restore_model_state = checkpoint['model_state']
        new_model_state = OrderedDict()
        for k, v in restore_model_state.items():
            if k[:4] == 'net.':
                name = k[4:]
            else:
                name = k
            new_model_state[name] = v
        self.load_state_dict(new_model_state, strict=True)
        logger.info('load stlt parameters !!!')
        # for name, param in self.named_parameters():
            # if 'classifier' not in name:
                # param.requires_grad = False

class VisReasonNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
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

        self.conv5 = nn.Conv3d(
            dim_in,
            self.cfg.VIT.EMBED_DIM,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )

        if cfg.TRAIN.MIXUP:
            self.obj_vis_embed = nn.Sequential(
                nn.Linear(
                    self.cfg.VIT.EMBED_DIM * self.crop_size[0] * self.crop_size[1],
                    self.cfg.VIT.EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(0.5))

            # self.mixer = mixup_helper.RandomMix(cfg)
            self.mixer = mixup_helper.PriorMix(cfg, k=1)

        if cfg.VIT.USE_POS == 'learned':
            self.pos_embed = vit_helper.PositionEmbeddingLearned(
                cfg.VIT.EMBED_DIM, cfg.VIT.ATTACH_POS
            )
        elif cfg.VIT.USE_POS == 'sine':
            self.pos_embed = vit_helper.PositionEmbeddingSine(
                cfg.VIT.EMBED_DIM // 3, attach=cfg.VIT.ATTACH_POS
            )
        else:
            self.pos_embed = nn.Identity()

        self.embed_dim = cfg.VIT.EMBED_DIM

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.reason_types = ['ho']
        self.vis_reason = HOReasonSimp(
            dim = self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            norm_layer=norm_layer,
            depths=cfg.REASON.VIS_DEPTHS,
            use_cls=cfg.MODEL.USE_CLS,
            pos_each_layer=cfg.VIT.POS_EACH_LAYER,
            reason_types=self.reason_types
        )

        self.head_drop = nn.Dropout(p=self.head_dropout)

        self.composition_classifier = nn.Linear(self.embed_dim, self.num_classes)

        self.vis_fusion = nn.Sequential(
            nn.Linear(len(self.reason_types) * self.embed_dim, self.embed_dim, bias=False),
            nn.ReLU(inplace=True))
        self.vis_classifier = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x, meta, labels=None, use_feature=False):
        if isinstance(x, list):
            x = x[0]
        B, C, T, H, W = x.shape
        box_input = meta['object_boxes'].clone().detach()
        roi_box_input = box_input[:, ::self.temporal_stride].contiguous() # b, T, nr_box, 4

        box_category = meta['box_category']
        box_category = box_category.long()
        # box_category = box_category.flatten()
        # box_category_embeddings = self.identity_embedding(box_category)
        # identity_repre = box_category_embeddings.view(
        #     B, -1, self.nr_boxes, self.embed_dim) # B, T(32), n, d

        x = self.conv5(x) # b,c,t,h,w
        # 首先把 feature map 给 token 化
        ## Add positional embeddings to input

        pos = self.pos_embed(x) # b,c,t,h,w
        x = torch.cat([x, pos], dim=1)

        # 用 roi 去得到 object token, hand token
        x = x.permute(0, 2, 3, 4, 1).contiguous() # b, t, h, w, c
        x = x.reshape(B*T, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        roi_box_input = roi_box_input.view(B*T, self.nr_boxes, 4)
        boxes_list = box_to_normalized(roi_box_input, crop_size=[224, 224])
        region_vis_feas = build_region_feas(x, boxes_list,
                                            self.crop_size, self.img_size, flatten=False) # b*T*nr_box, d, 3, 3
        region_vis_feas = region_vis_feas.permute(0, 2, 3, 1).contiguous() # b*t*nr_box, 3, 3, d

        region_vis_feas, roi_pos = region_vis_feas.chunk(2, dim=-1) # b*t*nr_box, 3, 3, d

        # prepare feature for mixup
        if self.training and self.cfg.TRAIN.MIXUP:
            obj_vis_feas = region_vis_feas.reshape(B, T, self.nr_boxes, -1) # b,t,n,d
            obj_vis_feas = self.obj_vis_embed(obj_vis_feas) # 降维
            obj_vis_feas = torch.mean(obj_vis_feas, dim=1) # b, n, d
            obj_indicator = meta['obj_indicator']

        region_vis_feas = region_vis_feas

        vis_tokens = region_vis_feas.reshape(B, T, self.nr_boxes, self.crop_size[0]*self.crop_size[1], -1)
        roi_pos = roi_pos.reshape(B, T, self.nr_boxes, self.crop_size[0]*self.crop_size[1], -1)

        # prepare hand/object tokens
        hand_vis_tokens = vis_tokens[:, :, :2] # b, t, 2, 3, 3, d
        obj_vis_tokens = vis_tokens[:, :, 2:] # b, t, n, 3, 3, d
        hand_pos = roi_pos[:, :, :2]
        obj_pos = roi_pos[:, :, 2:]
        hand_vis_tokens = hand_vis_tokens.reshape(B, -1, self.embed_dim)
        obj_vis_tokens = obj_vis_tokens.reshape(B, -1, self.embed_dim)
        hand_pos = hand_pos.reshape(B, -1, self.embed_dim)
        obj_pos = obj_pos.reshape(B, -1, self.embed_dim)

        vis_reason_outs = self.vis_reason(hand_vis_tokens, obj_vis_tokens, hand_pos, obj_pos)

        # ho, hh, oh, oo = vis_reason_outs
        if not self.cfg.MODEL.USE_CLS:
            vis_reason_outs = [torch.mean(out, dim=1) for out in vis_reason_outs] # each (b, 1, d)
        vis_reason_outs = torch.cat(vis_reason_outs, dim=-1).squeeze()
        vis_reason_outs = self.vis_fusion(vis_reason_outs)

        if self.training and self.cfg.TRAIN.MIXUP:
            ori_comp, mix_fea, mix_labels = self.mixer(obj_vis_feas, meta, labels)
            label_onehot = F.one_hot(labels.detach(), num_classes=self.num_classes)
            # mix_labels = label_onehot
            label_onehot = label_onehot.float()
            comp_cls = self.composition_classifier(self.head_drop(ori_comp))
            with torch.no_grad():
                pred = self.composition_classifier(self.head_drop(mix_fea.detach())) # b, k, 174
                y_hat = torch.matmul(pred, label_onehot.unsqueeze(-1)).squeeze(dim=-1) # b,k
                idx_best_comp = torch.argmax(y_hat, dim=1) # b
            select_noun_fea = mix_fea[torch.arange(B), idx_best_comp]
            mix_cls = self.composition_classifier(self.head_drop(select_noun_fea))
            mix_labels = mix_labels[torch.arange(B), idx_best_comp]

        if self.training and self.cfg.TRAIN.MIXUP:
            vis_cls = self.vis_classifier(self.head_drop(vis_reason_outs))
            outs = [vis_cls, vis_reason_outs] if use_feature else [vis_cls]
            return outs + [comp_cls, mix_cls, mix_labels]
        else:
            vis_cls = self.vis_classifier(self.head_drop(vis_reason_outs))
            if use_feature:
                return vis_cls, vis_reason_outs
            else:
                return vis_cls

    def get_mixed_obj(self, obj_vis_fea, obj_indicator, labels):
        '''
        obj_vis_fea: (b, n, d)
        obj_indicator: (b, n)

        return: (b, k, d)
        '''
        B, nr_box, _ = obj_vis_fea.shape
        k = 3
        label_onehot = F.one_hot(labels, num_classes=self.num_classes)
        batch_cand_fea_list = []
        batch_mix_label_list = []
        for i in range(B):
            cand_obj_fea_list = []
            cand_mix_label_list = []
            vid_obj_fea = obj_vis_fea[i].clone() # (n, d), 当前 video 的 obj feature.
            cand_idx = obj_indicator.clone().detach()
            cand_idx[i] = 0
            cand_idx = torch.nonzero(cand_idx != 0, as_tuple=True) # 候选的 obj index.
            # logger.info(cand_idx)
            cand_obj_num = cand_idx[0].size(0)
            ori_obj_num = torch.nonzero(obj_indicator[i] != 0, as_tuple=True)[0].size(0)

            if cand_obj_num == 0 or ori_obj_num == 0:
                batch_cand_fea_list.append(torch.stack([torch.mean(vid_obj_fea, dim=0)]*k, dim=0))
                batch_mix_label_list.append(torch.stack([label_onehot[i]]*k, dim=0))
                continue

            obj_prior_dis = torch.distributions.categorical.Categorical(probs=torch.ones(cand_obj_num))
            select_obj_idx = obj_prior_dis.sample(sample_shape=(k, ))
            lam = np.random.beta(1.0, 1.0)
            for j in range(k):
                ori_obj_slot = torch.randint(self.nr_boxes, (1, ))
                batch_idx, box_idx = cand_idx[0][select_obj_idx[j]], cand_idx[1][select_obj_idx[j]]
                select_obj_fea = obj_vis_fea[batch_idx, box_idx]
                select_vid_label = label_onehot[batch_idx]
                mixed_fea = vid_obj_fea[ori_obj_slot] * lam + select_obj_fea * (1-lam) # n, dim
                cand_obj_fea_list.append(torch.mean(mixed_fea, dim=0))
                cand_mix_label_list.append(label_onehot[i] * lam + select_vid_label * (1-lam))
            cand_obj_fea = torch.stack(cand_obj_fea_list, dim=0) # k,d
            cand_mix_label = torch.stack(cand_mix_label_list, dim=0) # k, 174

            batch_cand_fea_list.append(cand_obj_fea)
            batch_mix_label_list.append(cand_mix_label)

        batch_cand_fea = torch.stack(batch_cand_fea_list, dim=0) # b, k, d
        batch_mix_label = torch.stack(batch_mix_label_list, dim=0) # b, k, 174

        return batch_cand_fea, batch_mix_label

