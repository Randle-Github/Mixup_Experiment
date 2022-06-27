from torchvision.ops.roi_align import roi_align
import torch

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
                      img_size=[224, 224],
                      flatten=True):
    # Building feas for each bounding box by using RoI Align
    # feature_maps:[N,C,H,W], where N=b*T
    IH, IW = img_size
    FH, FW = feature_maps.size()[-2:]  # Feature_H, Feature_W
    region_feas = roi_align(feature_maps,
                            boxes_list,
                            output_crop_size,
                            spatial_scale=float(FW) /
                            IW)  # b*T*K, C, S, S; S denotes output_size
    if flatten:
        return region_feas.view(region_feas.size(0), -1)  # b*T*K, D*S*S
    else:
        return region_feas # b*T*nr_box, d, 3, 3

def slice_bbox(boxes_tensor, crop_size, img_size):
    # boxes_tensor: (b, T, n, 4), cx, cy, w, h
    # crop size: int, typically 3
    # return (b, T, n, crop_size^2, 4)
    B, T, nr_box, _ = boxes_tensor.shape
    new_boxes_tensor = boxes_tensor.clone()
    new_boxes_tensor = new_boxes_tensor.repeat(1, 1, 1, crop_size*crop_size) # b,t,n,crop*crop*4

    tl_x = boxes_tensor[..., 0] - boxes_tensor[..., 2] / 2.0 # b,t,n
    tl_y = boxes_tensor[..., 1] - boxes_tensor[..., 3] / 2.0 # b,t,n

    slice_bw, slice_bh = boxes_tensor[..., 2] / crop_size, boxes_tensor[..., 3] / crop_size # b,t,n

    for i in range(crop_size*crop_size):
        new_boxes_tensor[..., i*4] = (tl_x + (i % crop_size * 2 + 1) * slice_bw / 2.0)
        new_boxes_tensor[..., i*4+1] = tl_y + (i // crop_size * 2 + 1) * slice_bh / 2.0
        new_boxes_tensor[..., i*4+2] = slice_bw
        new_boxes_tensor[..., i*4+3] = slice_bh
    new_boxes_tensor = new_boxes_tensor.reshape(B, T, nr_box, crop_size*crop_size, 4)

    return new_boxes_tensor
    

    
    
