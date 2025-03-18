# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# global alignment optimization wrapper function
# --------------------------------------------------------
from enum import Enum

from .optimizer import PointCloudOptimizer
from .modular_optimizer import ModularPointCloudOptimizer
from .pair_viewer import PairViewer


class GlobalAlignerMode(Enum):
    PointCloudOptimizer = "PointCloudOptimizer"
    ModularPointCloudOptimizer = "ModularPointCloudOptimizer"
    PairViewer = "PairViewer"

def create_idx_to_image_mapping(idx_list, image_name_list):
    idx_to_image = {}
    for idx, img_name in zip(idx_list, image_name_list):
        if idx not in idx_to_image:
            idx_to_image[idx] = set()
        idx_to_image[idx].add(img_name)
    return idx_to_image

def global_aligner(dust3r_output, device, mode=GlobalAlignerMode.PointCloudOptimizer, **optim_kw):
    # extract all inputs
    view1, view2, pred1, pred2 = [dust3r_output[k] for k in 'view1 view2 pred1 pred2'.split()]
    view1_mapping = create_idx_to_image_mapping(view1['idx'], view1['image_name'])
    view2_mapping = create_idx_to_image_mapping(view2['idx'], view2['image_name'])
    assert view1_mapping==view2_mapping
    view_mapping = view1_mapping
    # build the optimizer
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        net = PointCloudOptimizer(view1, view2, pred1, pred2, view_mapping,**optim_kw).to(device)
    elif mode == GlobalAlignerMode.ModularPointCloudOptimizer:
        net = ModularPointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    elif mode == GlobalAlignerMode.PairViewer:
        net = PairViewer(view1, view2, pred1, pred2, **optim_kw).to(device)
    else:
        raise NotImplementedError(f'Unknown mode {mode}')

    return net
