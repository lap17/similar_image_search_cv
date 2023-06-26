import sys
import gc
import os
from typing import List
from typing import List, NamedTuple, Optional
import cv2
import base64
import torch
import numpy as np
import pandas as pd
import pytz
import datetime
from PIL import Image
import time
import threading
from os import listdir
from os.path import isfile, join

confidence_threshold = 0.3

CLASSES = [ 'short sleeve top', 'long sleeve top', 'short sleeve outwear', 
            'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 
            'skirt', 'short sleeve dress', 'long sleeve dress',
            'vest dress', 'sling dress', 'dress', 'handbag', 'swimwear']


group_handbag = ['handbag']
group_swimwear = ['swimwear']
group_bottom = ['trousers', 'skirt', 'shorts']
group_upper = ['dress', 'long sleeve dress', 'vest dress', 'sling dress', 'short sleeve dress', 'short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling']

dict_class_to_group = {
    'short sleeve top': 'upper',
    'long sleeve top': 'upper',
    'short sleeve outwear': 'upper',
    'long sleeve outwear': 'upper',
    'vest': 'upper',
    'sling': 'upper',
    'shorts': 'bottom',
    'trousers': 'bottom',
    'skirt': 'bottom',
    'short sleeve dress': 'upper',
    'long sleeve dress': 'upper',
    'vest dress': 'upper',
    'sling dress': 'upper',
    'dress': 'upper',
    'handbag': 'handbag',
    'swimwear': 'swimwear'
}


def get_yolo5():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='yolo5_model.pt')


model = get_yolo5()


def get_preds(model, img):
    return model([img]).xyxy[0].cpu().numpy()


def crop_image(path_orig_images, path_catalog, filename, model):
    img = cv2.imread(path_orig_images + '/' + filename, cv2.IMREAD_COLOR)
    filename_without_type = filename.rsplit('.',1)[0]
    img_for_pred = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = get_preds(model, img_for_pred )
    result_copy = result.copy()
    #result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
    j = 0
    for bbox_data in result_copy:
        j+=1
        xmin, ymin, xmax, ymax, conf, label = bbox_data
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        label = int(label)
        class_name = CLASSES[label]
        if conf > confidence_threshold:
            group = dict_class_to_group[class_name]
            cv2.imwrite(path_catalog + '/' + group + '/' + filename_without_type + '_SEPSP_crop_' + str(j) + '.jpg', img[ymin:ymax, xmin:xmax])

 
def detect_and_save_crop_images(catalog_name):
    path_catalog = 'catalogs/' + catalog_name
    path_orig_images = path_catalog + '/original_images'
    files = [f for f in listdir(path_orig_images) if (isfile(join(path_orig_images, f)))]
    for filename in files:
        crop_image(path_orig_images, path_catalog, filename, model)


def get_crops_and_classes(img_orig):
    #model = get_yolo5()
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    result = get_preds(model, img)
    result_copy = result.copy()
    list_cropped_images = []
    for bbox_data in result_copy:
        xmin, ymin, xmax, ymax, conf, label = bbox_data
        conf = round(conf, 2)
        if conf >= confidence_threshold:
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            label = int(label)
            class_name = CLASSES[label]
            cropped_image = img[ymin:ymax, xmin:xmax]
            w, h, t = cropped_image.shape
            list_cropped_images.append([w + h, cropped_image, class_name])
    if len(list_cropped_images)!=0:
        list_cropped_images = sorted(list_cropped_images, reverse=True, key=lambda x: x[0])
        list_cropped_images = list_cropped_images[:3]
    return list_cropped_images
