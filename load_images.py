import zipfile
import os
from os import listdir
from os.path import isfile, join
import glob
import shutil
import boto3

ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

list_groups = ['handbag', 'swimwear', 'bottom', 'upper']


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    s3.upload_file(local_file, bucket, s3_file)


def check_created_catalog(name_catalog):
    list_catalogs = os.listdir('catalogs')
    if name_catalog not in list_catalogs:
        main_path = 'catalogs/' + name_catalog
        os.mkdir(main_path)
        os.mkdir(main_path + '/original_images')
        for group in list_groups:
            os.mkdir(main_path + '/' + group)


def load_images_to_calalog(name_catalog, file_path):
    path_for_unzipped = 'upload_folder/' + name_catalog
    type_file = file_path.rsplit('.',1)[-1]
    if type_file!='zip':
        os.remove(file_path)
        return 'Uploaded file is not in zip type'
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall('upload_folder/' + name_catalog)
    check_created_catalog(name_catalog)
    path_to_save = 'catalogs/'+ name_catalog
    paths = [ path_for_unzipped + '/*/*/*.jpg', path_for_unzipped+ '/*/*/*.png',
              path_for_unzipped+ '/*/*.jpg', path_for_unzipped+ '/*/*.png', 
              path_for_unzipped+ '/*.jpg', path_for_unzipped+ '/*.png']
    list_all_images = []
    for path in paths:
        list_images = glob.glob(path)
        list_all_images = list_all_images + list_images
    if len(list_all_images) == 0:
        shutil.rmtree(path_for_unzipped)
        os.remove(file_path)
        return 'No images found'
    for image_path in list_all_images:
        shutil.copy(image_path, path_to_save + '/original_images')
        filename = os.path.basename(image_path)
        upload_to_aws(image_path, 'shopparty', name_catalog+filename)
    shutil.rmtree(path_for_unzipped)
    os.remove(file_path)
    return str(len(list_all_images)) + ' images uploaded to catalog'
