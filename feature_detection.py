from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2

from superglue_models.matching import Matching
from superglue_models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image, 
                          read_image_v2, rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }

matching = Matching(config).eval().to(device)


def get_most_similar_image(img, list_similar_images):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    image0, inp0 = read_image_v2(img, device)
    for path_similar_image in list_similar_images:
        img1 = cv2.imread(path_similar_image, cv2.IMREAD_GRAYSCALE)
        image1, inp1 = read_image_v2(img1, device)
        pred = matching({'image0': inp0, 'image1': inp1})
        try:
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        except:
            pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        if len(mkpts0) > 50:
            return path_similar_image
    return None
