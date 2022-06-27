import math
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import compaction.utils.logging as logging

logger = logging.get_logger(__name__)

def mixup_process(out, target, alpha=1.0, num_classes=174):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    # lam = 0.5
    # adjcent mixup ###########################################################
    # indices_even = torch.arange(0, B, step=2)
    # indices_odd = torch.arange(1, B, step=2)
    # indices_l = indices_even
    # indices_r = indices_odd
    indices = np.random.permutation(out.size(0))    
    target = target.long()
    out = out*lam + out[indices]*(1-lam)
    target_onehot = F.one_hot(target, num_classes=num_classes).to(device=target.device)
    target_reweighted = target_onehot * lam + target_onehot[indices] * (1 - lam)
    return out, target_reweighted

def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def token_mix(x, target, num_frame, num_candi=1, alpha=1.0, num_classes=174):
    # x: b, patches, d
    # cut-mix like. 随机取一个区域 (Txobj), 进行拼接
    B, len_token, d = x.shape
    lam = np.random.beta(alpha, alpha)
    num_spatial = len_token // num_frame
    x = x.reshape(B, num_frame, -1, d)

    target = target.long()
    target_onehot = F.one_hot(target, num_classes=num_classes)
    candi_fea_list = []
    for i in range(num_candi):
        indices = np.random.permutation(x.size(0))
        bbx1, bby1, bbx2, bby2 = rand_bbox(num_frame, num_spatial, lam)
        temp_x = x.clone()
        temp_x[:, bbx1:bbx2, bby1:bby2] = x[indices, bbx1:bbx2, bby1:bby2]
        temp_x = temp_x.reshape(B, -1, d).unsqueeze(1) # b,1,patches, d
        candi_fea_list.append(temp_x)
    # target_shuffled_onehot = target_onehot[indices]
    # target_reweighted = target_onehot * lam + target_shuffled_onehot * (1 - lam)
    candi_fea = torch.stack(candi_fea_list, dim=1) # b, k, patches, d

    return candi_fea

class InputMixup():
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x, boxes, meta, target):
        """TODO describe function

        :param x: b, 3, t, h, w
        :param boxes: b, t, n, 4
        :param meta: 
        :param target: b,  
        :returns: 

        """
        
        B = x.shape[0]
        x = x.permute(0, 2, 3, 4, 1).contiguous() # b, t, h, w, 3
        swap_obj_boxes = boxes[:, :, 2] # b, t, 4
        swap_obj_boxes = self._box_to_normalize(swap_obj_boxes) # b, t, 4
        

    def _box_to_normalize(self, boxes_tensor, crop_size=(224, 224)):
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
        return new_boxes_tensor

class PriorMix(nn.Module):
    '''
    随机从其他 video 中选取 object, 然后 mix 到当前 video.
    '''
    def __init__(self, cfg, k=3) -> None:
        super().__init__()
        self.k = k # 候选组合数目
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.nr_boxes = cfg.DATA.NUM_BOXES
        self.cooccurrence = torch.from_numpy(np.load('./ckpt/val_coc.npy')).float() # (174, 320)
        pad = torch.zeros(174, 1)
        self.cooccurrence = torch.cat([pad, self.cooccurrence], dim=-1).cuda() # (174, 321)

    def forward(self, obj_fea, meta, labels):
        '''
        inter_fea: not used
        obj_fea: (b, n, d)
        obj_indicator: (b, n)
        '''
        B, nr_box, _ = obj_fea.shape
        nr_obj = nr_box - 2
        obj_indicator = meta['obj_indicator'][:, 2:]
        obj_categories = meta['obj_category'][:, 2:]
        batch_obj_cat_one_hot = F.one_hot(obj_categories, num_classes=321)  # (b, n, 321)
        
        ori_comp = torch.mean(obj_fea, dim=1)
        hand_fea, obj_fea = torch.split(obj_fea, [2, nr_obj], dim=1)
        label_onehot = F.one_hot(labels, num_classes=self.num_classes)
        mix_comp_list = []
        mix_label_list = []
        for i in range(B):
            cand_obj_fea_list = []
            cand_mix_label_list = []
            
            # label_one_hot = F.one_hot(labels[i], num_classes=self.num_classes).float()  # (1, 174) # yapf: disable
            # obj_occurrence_vec = torch.matmul(
                # label_one_hot.unsqueeze(0),
                # self.cooccurrence.to(device=label_one_hot.device))  # (1,174)*(174, 321), (1, 321)
            obj_occurrence_vec = self.cooccurrence[labels[i]] # (1, 321)
            cand_obj = (obj_occurrence_vec * batch_obj_cat_one_hot).sum(
                dim=-1)  # (b, n), 不为 0 的 entry 说明这个物体可以和这个 verb 组合
            cand_obj[i] = 0
            cand_idx = torch.nonzero(cand_obj != 0, as_tuple=True) # 候选的 obj index.
            cand_obj_num = cand_idx[0].size(0)
            ori_obj_num = torch.nonzero(obj_indicator[i] != 0, as_tuple=True)[0].size(0)
            vid_obj_fea = obj_fea[i].clone() # (n, d), 当前 video 的 obj feature.

            if cand_obj_num == 0 or ori_obj_num == 0:
                mix_comp_list.append(torch.stack([torch.mean(vid_obj_fea, dim=0)]*self.k, dim=0))
                mix_label_list.append(torch.stack([label_onehot[i]]*self.k, dim=0))
                continue

            obj_candidates_prob = torch.log(cand_obj[cand_idx])
            obj_candidates_prob = F.softmax(obj_candidates_prob, dim=0)
            # obj_candidates_prob = cand_obj[cand_idx]
            obj_prior_dis = torch.distributions.categorical.Categorical(probs=obj_candidates_prob)
            select_obj_idx = obj_prior_dis.sample(sample_shape=(self.k, ))
            lam = np.random.beta(1.0, 1.0)
            for j in range(self.k):
                ori_obj_slot = torch.randint(nr_obj, (1, ))
                batch_idx, box_idx = cand_idx[0][select_obj_idx[j]], cand_idx[1][select_obj_idx[j]]
                select_obj_fea = obj_fea[batch_idx, box_idx]
                select_vid_label = label_onehot[batch_idx]
                mixed_fea = vid_obj_fea[ori_obj_slot] * lam + select_obj_fea * (1-lam) # 1, dim
                # vid_obj_fea[ori_obj_slot] = mixed_fea
                # cand_obj_fea_list.append(torch.mean(vid_obj_fea, dim=0))
                cand_obj_fea_list.append(torch.mean(mixed_fea, dim=0))
                # cand_obj_fea_list.append(torch.mean(vid_obj_fea, dim=0))
                cand_mix_label_list.append(label_onehot[i] * lam + select_vid_label * (1-lam))
            cand_obj_fea = torch.stack(cand_obj_fea_list, dim=0) # k,d
            cand_mix_label = torch.stack(cand_mix_label_list, dim=0) # k, 174
            cand_prob = obj_candidates_prob[select_obj_idx] # k

            mix_comp_list.append(cand_obj_fea)
            mix_label_list.append(cand_mix_label)

        mix_fea = torch.stack(mix_comp_list, dim=0) # b, k, d
        mix_label = torch.stack(mix_label_list, dim=0) # b, k, 174

        return ori_comp, mix_fea, mix_label

class TokenMixer(nn.Module):
    def __init__(self, cfg) -> None:
        super(TokenMixer, self).__init__()
        self.k = 3
        self.num_classes = 174
        self.embed_dim = cfg.VIT.EMBED_DIM

    def forward(self, region_vis_feas, obj_indicator, labels):
        '''
        region_vis_feas: b, t, n, 3*3, d
        obj_indicator: b, N
        '''
        B, T, nr_box, _, d = region_vis_feas.shape
        obj_indicator = obj_indicator[:, 2:] # 取 object, b, n
        region_vis_feas = region_vis_feas.transpose(1, 2) # b, n, t, 3*3, d
        mix_comp_list = []
        mix_label_list = []
        label_onehot = F.one_hot(labels, num_classes=self.num_classes)
        for i in range(B):
            cand_obj_fea_list = []
            cand_mix_label_list = []
            vid_region_vis_feas = region_vis_feas[i].clone() # N, t, 3*3, d
            cand_idx = obj_indicator.clone().detach()
            cand_idx[i] = 0
            cand_idx = torch.nonzero(cand_idx != 0, as_tuple=True) # 候选的 obj index.
            cand_obj_num = cand_idx[0].size(0)
            ori_obj_num = torch.nonzero(obj_indicator[i] != 0, as_tuple=True)[0].size(0)

            if cand_obj_num == 0 or ori_obj_num == 0:
                mix_comp_list.append(torch.stack([vid_region_vis_feas]*self.k, dim=0))
                mix_label_list.append(torch.stack([label_onehot[i]]*self.k, dim=0))
                continue

            obj_prior_dis = torch.distributions.categorical.Categorical(probs=torch.ones(cand_obj_num))

            select_obj_idx = obj_prior_dis.sample(sample_shape=(self.k, ))
            lam = np.random.beta(1.0, 1.0)

            for j in range(self.k):
                ori_obj_slot = torch.randint(nr_box, (1, ))
                batch_idx, box_idx = cand_idx[0][select_obj_idx[j]], cand_idx[1][select_obj_idx[j]]
                select_obj_fea = region_vis_feas[batch_idx, box_idx] # t, 3*3, d
                select_vid_label = label_onehot[batch_idx]
                mixed_fea = vid_region_vis_feas[ori_obj_slot] * lam + select_obj_fea * (1-lam) # t, 3*3, d
                vid_region_vis_feas[ori_obj_slot] = mixed_fea # N, t, 3*3, d

                cand_obj_fea_list.append(vid_region_vis_feas)
                cand_mix_label_list.append(label_onehot[i] * lam + select_vid_label * (1-lam))

            cand_obj_fea = torch.stack(cand_obj_fea_list, dim=0) # k, N, t, 3*3, d
            cand_mix_label = torch.stack(cand_mix_label_list, dim=0) # k, 174

            mix_comp_list.append(cand_obj_fea)
            mix_label_list.append(cand_mix_label)

        mix_obj = torch.stack(mix_comp_list, dim=0) # b, k, n, t, 3*3, d
        mix_obj = mix_obj.transpose(2, 3).contiguous() # b, k, t, n, 3*3, d
        mix_label = torch.stack(mix_label_list, dim=0) # b, k, 174

        return mix_obj, mix_label

class RandomMix(nn.Module):
    '''
    随机从其他 video 中选取 object, 然后 mix 到当前 video.
    '''
    def __init__(self, cfg, k=3) -> None:
        super().__init__()
        self.k = k # 候选组合数目
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.nr_boxes = cfg.DATA.NUM_BOXES

    def forward(self, obj_fea, obj_indicator, labels):
        '''
        inter_fea: not used
        obj_fea: (b, n, d)
        obj_indicator: (b, n)
        '''
        B, nr_box, _ = obj_fea.shape
        ori_comp = torch.mean(obj_fea, dim=1)
        label_onehot = F.one_hot(labels, num_classes=self.num_classes)
        mix_comp_list = []
        mix_label_list = []
        for i in range(B):
            cand_obj_fea_list = []
            cand_mix_label_list = []
            vid_obj_fea = obj_fea[i].clone() # (n, d), 当前 video 的 obj feature.
            cand_idx = obj_indicator.clone().detach()
            cand_idx[i] = 0
            cand_idx = torch.nonzero(cand_idx != 0, as_tuple=True) # 候选的 obj index.
            cand_obj_num = cand_idx[0].size(0)
            ori_obj_num = torch.nonzero(obj_indicator[i] != 0, as_tuple=True)[0].size(0)

            if cand_obj_num == 0 or ori_obj_num == 0:
                mix_comp_list.append(torch.stack([torch.mean(vid_obj_fea, dim=0)]*self.k, dim=0))
                mix_label_list.append(torch.stack([label_onehot[i]]*self.k, dim=0))
                continue

            obj_prior_dis = torch.distributions.categorical.Categorical(probs=torch.ones(cand_obj_num))
            select_obj_idx = obj_prior_dis.sample(sample_shape=(self.k, ))
            lam = np.random.beta(1.0, 1.0)
            for j in range(self.k):
                ori_obj_slot = torch.randint(nr_box, (1, ))
                # ori_obj_slot = 2
                batch_idx, box_idx = cand_idx[0][select_obj_idx[j]], cand_idx[1][select_obj_idx[j]]
                select_obj_fea = obj_fea[batch_idx, box_idx]
                select_vid_label = label_onehot[batch_idx]
                mixed_fea = vid_obj_fea[ori_obj_slot] * lam + select_obj_fea * (1-lam) # 1, dim
                # vid_obj_fea[ori_obj_slot] = mixed_fea # n, d
                # cand_obj_fea_list.append(torch.mean(vid_obj_fea, dim=0))
                cand_obj_fea_list.append(torch.mean(mixed_fea, dim=0))
                cand_mix_label_list.append(label_onehot[i] * lam + select_vid_label * (1-lam))
            cand_obj_fea = torch.stack(cand_obj_fea_list, dim=0) # k,d
            cand_mix_label = torch.stack(cand_mix_label_list, dim=0) # k, 174

            mix_comp_list.append(cand_obj_fea)
            mix_label_list.append(cand_mix_label)

        mix_fea = torch.stack(mix_comp_list, dim=0) # b, k, d
        mix_label = torch.stack(mix_label_list, dim=0) # b, k, 174

        return ori_comp, mix_fea, mix_label

class MixUp:
    """
    Apply mixup and/or cutmix for videos at batch level.
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
        Features (https://arxiv.org/abs/1905.04899)
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        mix_prob=1.0,
        switch_prob=0.5,
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=1000,
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            cutmix_alpha (float): Cutmix alpha value.
            mix_prob (float): Probability of applying mixup or cutmix.
            switch_prob (float): Probability of switching to cutmix instead of
                mixup when both are active.
            correct_lam (bool): Apply lambda correction when cutmix bbox
                clipped by image borders.
            label_smoothing (float): Apply label smoothing to the mixed target
                tensor. If label_smoothing is not used, set it to 0.
            num_classes (int): Number of classes for target.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.correct_lam = correct_lam

    def _get_mixup_params(self):
        lam = 1.0
        use_cutmix = False
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._get_mixup_params()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = get_cutmix_bbox(
                x.shape,
                lam,
                correct_lam=self.correct_lam,
            )
            x[..., yl:yh, xl:xh] = x.flip(0)[..., yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        assert len(x) > 1, "Batch size should be greater than 1 for mixup."
        lam = self._mix_batch(x)
        target = mixup_target(
            target, self.num_classes, lam, self.label_smoothing
        )
        return x, target
    

def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        on_value (float): Target Value for ground truth class.
        off_value (float): Target Value for other classes.This value is used for
            label smoothing.
    """

    targets = targets.long().view(-1, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value)


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixup/cutmix.
        smoothing (float): Label smoothing value.
    """
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    target1 = convert_to_one_hot(
        target,
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    target2 = convert_to_one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    return target1 * lam + target2 * (1.0 - lam)    


def get_cutmix_bbox(img_shape, lam, correct_lam=True, count=None):
    """
    Generates the box coordinates for cutmix.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        correct_lam (bool): Apply lambda correction when cutmix bbox clipped by
            image borders.
        count (int): Number of bbox to generate
    """

    yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam
