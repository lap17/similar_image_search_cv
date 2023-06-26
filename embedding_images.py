import click
import requests
from io import BytesIO
from pathlib import Path
import pickle
from PIL import Image as pil_img
import numpy as np

from fastai.vision.data import ImageDataBunch
from fastai.vision.transform import get_transforms
from fastai.vision.learner import create_cnn, cnn_learner
from fastai.vision import models
from fastai.vision.image import pil2tensor, Image

import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import os
from imutils import resize
from lshash import LSHash
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

classes = [ 'short sleeve top', 'long sleeve top', 'short sleeve outwear', 
            'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 
            'skirt', 'short sleeve dress', 'long sleeve dress',
            'vest dress', 'sling dress', 'dress', 'handbag', 'swimwear']

list_groups = ['handbag', 'swimwear', 'bottom', 'upper']


def load_image_databunch(input_path, classes):
    tfms = get_transforms(
        do_flip=False,
        flip_vert=False,
        max_rotate=0,
        max_lighting=0,
        max_zoom=0,
        max_warp=0,
    )

    data_bunch = ImageDataBunch.single_from_classes(
        Path(input_path), classes, ds_tfms=tfms, size=250
    )
    return data_bunch


def load_model(data_bunch, model_type, model_name):
    learn = cnn_learner(data_bunch, model_type, pretrained=False)
    learn.load(model_name)
    return learn


class SaveFeatures:
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))

    def remove(self):
        self.hook.remove()


def image_to_vec(url_img, hook, learner):
    _ = learner.predict(Image(pil2tensor(url_img, np.float32).div_(255)))
    vect = hook.features[-1]
    return vect


def get_vect(url_img, conv_learn, hook):
    vect = image_to_vec(url_img, hook, conv_learn)
    return vect


def resized_img_by_height(im):
    w, h = im.size
    resized_height = 250
    wpercent = (resized_height/float(h))
    resized_width = int((float(w)*float(wpercent)))
    im = im.resize((resized_width, resized_height), pil_img.ANTIALIAS)
    #im = im.convert('RGB')
    return im


def resized_img_by_width(im):
    w, h = im.size
    resized_width = 250
    hpercent = (resized_width/float(w))
    resized_height = int((float(h)*float(hpercent)))
    im = im.resize((resized_width, resized_height), pil_img.ANTIALIAS)
    #im = im.convert('RGB')
    return im


def get_clothes_embedding(catalog_name):
    k = 10 
    L = 5  
    d = 512 
    data_bunch = load_image_databunch("resnet_models", classes)
    learner = load_model(data_bunch, models.resnet34, "stg1-rn34")
    sf = SaveFeatures(learner.model[1][5])
    path_catalog = 'catalogs/' + catalog_name
    for group in list_groups:
        lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)
        path_group = path_catalog + '/' + group
        dirs = os.listdir(path_group)
        if len(dirs) == 0:
            continue
        dict_res = {}
        for file_name in dirs:
            im = pil_img.open(path_group + '/' + file_name).convert('RGB')
            image_width, image_height = im.size
            if image_width > image_height:
                im = resized_img_by_width(im)
            else:
                im = resized_img_by_height(im)
            vect = get_vect(im , learner, sf)
            dict_res[path_catalog + '/' + group + '/' + file_name ] = vect
        for img_path, vec in dict_res.items():
            lsh.index(vec.flatten(), extra_data=img_path)
        pickle.dump(lsh, open(path_catalog + '/lsh_'+ group + '.p', "wb"))


def get_list_similar_images(pil_img, class_name, name_catalog, group):
    image_width, image_height = pil_img.size
    if image_width > image_height:
        pil_img = resized_img_by_width(pil_img)
    else:
        pil_img = resized_img_by_height(pil_img)
    path_to_lsh = 'catalogs/'+name_catalog+'/lsh_' + str(group) + '.p'
    if not os.path.exists(path_to_lsh):
        return []
    data_bunch = load_image_databunch("resnet_models", classes)
    learner = load_model(data_bunch, models.resnet34, "stg1-rn34")
    sf = SaveFeatures(learner.model[1][5])
    lsh = pickle.load(open(path_to_lsh, "rb"))
    vect = get_vect(pil_img, learner, sf)
    response = lsh.query(vect, num_results = 5, distance_func="hamming")
    list_images = []
    for res in response:
        list_images.append(res[0][1])
    return list_images
