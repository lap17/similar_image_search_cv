from flask import Flask
import glob
from flask import request
from flask import jsonify, safe_join, abort, send_file
import json
from flask import Response
from werkzeug.utils import secure_filename
import zipfile
import os
from os import listdir
from os.path import isfile, join
from crop_images import detect_and_save_crop_images, get_crops_and_classes
from load_images import load_images_to_calalog
from embedding_images import get_clothes_embedding, get_list_similar_images
import shutil
import random
import io
import PIL.Image as pil_img
import cv2
from feature_detection import get_most_similar_image
import boto3

ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

ALLOWED_EXTENSIONS = {'png', 'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload_folder'

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


def delete_from_aws(s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    s3.delete_object(Bucket='shopparty', Key=s3_file)


def get_url_from_aws(s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    url = s3.generate_presigned_url( ClientMethod='get_object',
                                     Params={
                                        'Bucket': 'shopparty',
                                        'Key': s3_file
                                     })
    return url


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resized_img_by_height_v2(im):
    w, h = im.size
    resized_height = 300
    wpercent = (resized_height/float(h))
    resized_width = int((float(w)*float(wpercent)))
    return resized_width, resized_height


def get_detected_list_items(name_catalog, file_path):
    img_orig = cv2.imread(file_path, cv2.IMREAD_COLOR)
    list_cropped_images = get_crops_and_classes(img_orig)
    if len(list_cropped_images)==0:
        return None, None
    list_items = []
    for cropped_image in list_cropped_images:
        class_name = cropped_image[2]
        group = dict_class_to_group[class_name]
        cropped_image_cv = cropped_image[1]
        cropped_image_pil = pil_img.fromarray(cropped_image_cv)
        list_similar_images = get_list_similar_images(cropped_image_pil, class_name, name_catalog, group)
        if len(list_similar_images) == 0:
            continue
        most_similar_image = get_most_similar_image(cropped_image_cv, list_similar_images)
        if most_similar_image!=None:
            filename = most_similar_image.rsplit('/')[-1]
            itemname = filename.split('_SEPSP_')[0]
            itemname_v1 = 'catalogs/' + name_catalog + '/original_images/' + itemname + '.jpg'
            itemname_v2 = 'catalogs/' + name_catalog + '/original_images/' + itemname + '.png'
            if os.path.exists(itemname_v1):
                list_items.append(itemname_v1)
            elif os.path.exists(itemname_v2):
                list_items.append(itemname_v2)
    return list_items


def get_detected_one_item(name_catalog, file_path):
    img_orig = cv2.imread(file_path, cv2.IMREAD_COLOR)
    list_cropped_images = get_crops_and_classes(img_orig)
    if len(list_cropped_images)==0:
        return None, None
    list_items = []
    cropped_image = list_cropped_images[0]
    class_name = cropped_image[2]
    group = dict_class_to_group[class_name]
    cropped_image_cv = cropped_image[1]
    cropped_image_pil = pil_img.fromarray(cropped_image_cv)
    cropped_image_pil.show()
    list_similar_images = get_list_similar_images(cropped_image_pil, class_name, name_catalog, group)
    if len(list_similar_images) == 0:
        return None, None
    most_similar_image = get_most_similar_image(cropped_image_cv, list_similar_images)
    if most_similar_image!=None:
        filename = most_similar_image.rsplit('/')[-1]
        itemname = filename.split('_SEPSP_')[0]
        itemname_v1 = 'catalogs/' + name_catalog + '/original_images/' + itemname + '.jpg'
        itemname_v2 = 'catalogs/' + name_catalog + '/original_images/' + itemname + '.png'
        if os.path.exists(itemname_v1):
            find_img = pil_img.open(itemname_v1)
        elif os.path.exists(itemname_v2):
            find_img = pil_img.open(itemname_v2)
        return find_img, filename
    else:
        return None, None


@app.route('/catalog/add_images/', methods=['POST'])
def _catalog_add_images_():
    name_catalog = request.form['name_catalog']
    catalogs = listdir('catalogs')
    if name_catalog in catalogs:
        return jsonify({'status': 400, 'message': 'Catalog ' + str(name_catalog) + ' is exist'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 400, 'message': 'No selected file'})
    filename = secure_filename(file.filename)
    file_path = safe_join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    mess = processing_images(name_catalog, file_path)
    if mess in ['Uploaded file is not in zip type', 'No images found']:
        return jsonify({'status': 400, 'message': mess})
    else:
        return jsonify({'status': 200, 'message': mess})


@app.route('/catalog/delete/', methods=['DELETE'])
def _catalog_delete_():
    name_catalog = request.form['name_catalog']
    catalogs = listdir('catalogs')
    if name_catalog not in catalogs:
        return jsonify({'status': 400, 'message': 'Catalog ' + str(name_catalog) + ' not exist'})
    else:
        paths = ['catalogs/' + str(name_catalog) + '/original_images/*.jpg', 'catalogs/' + str(name_catalog) + '/original_images/*.png']
        list_all_images = []
        for path in paths:
            list_images = glob.glob(path)
            list_all_images = list_all_images + list_images
        for pathimage in list_all_images:
            filename = os.path.basename(pathimage)
            delete_from_aws(name_catalog+filename)
        shutil.rmtree('catalogs/' + str(name_catalog))
        return jsonify({'status': 200, 'message': 'Catalog ' + str(name_catalog) + ' deleted successfully'})


def processing_images(name_catalog, file_path):
    res = load_images_to_calalog(name_catalog, file_path)
    if res in ['Uploaded file is not in zip type', 'No images found']:
        return res
    detect_and_save_crop_images(name_catalog)
    get_clothes_embedding(name_catalog)
    return res


@app.route('/detect_items/', methods=['POST'])
def _detect_items_():
    name_catalog = request.form['name_catalog']
    catalogs = listdir('catalogs')
    if name_catalog not in catalogs:
        return jsonify({'status': 400, 'message': 'Catalog ' + str(name_catalog) + ' not exist'})
    file = request.files['file']
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({'status': 400, 'message': 'Uploaded file is not in png or jpg type'})
    new_filename = str(random.randint(10000, 99999)) + '_' +  filename
    file_path = safe_join(app.config['UPLOAD_FOLDER'], new_filename)
    file.save(file_path)
    list_items = get_detected_list_items(name_catalog, file_path)
    os.unlink(file_path)
    if len(list_items) == 0:
        return jsonify({'status': 200, 'message': 'Item not found from catalog', 'list_items': []})
    list_urls = []
    for item in list_items:
        filename = os.path.basename(item)
        url = get_url_from_aws(name_catalog+filename)
        list_urls.append(url)
    return jsonify({'status': 200, 'message': 'Found similar items from the catalog', 'list_items': list_urls})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
