import argparse
import os
import shutil
import cv2
import tensorflow as tf
from tensorflow.keras.processing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# region Argument Parsing

parser = argparse.ArgumentParser(
    prog=f'python train_model.py',
    description='Training model for Magic Mirror prop used for Lockwood Immersive\'s HES Ball 2025 integration.',
    epilog='Created by Than | https://github.com/thana-than/LockwoodImmersiveMagicMirror')
parser.add_argument('-g', '--generate', default='', help='Generate extrapolated training data from folder.')
parser.add_argument('-d', '--directory', default='training_data/gen', help='Directory of training data.')
parser.add_argument('-o', '--output', default='training_data/model.h5', help='Location to output model file.')
parser.add_argument('-s', '--skip-training', action='store_true', default=False, help='Skip model training step.')
parser.add_argument('-t', '--transformations', default=64, help='How many transformations to generate per image.')
args = parser.parse_args()

# endregion

# region Generate Data

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

def generate_data(in_dir, out_dir, transformations):
    print(f'Generating extrapolated data from {in_dir} to {out_dir}...')
    #* Clear output directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for path, subdirs, filenames in os.walk(in_dir):
        out_dir_equivalent = os.path.join(out_dir, os.path.relpath(path, in_dir))
        print(f'Processing folder: {path} out {out_dir_equivalent}')
        os.makedirs(out_dir_equivalent, exist_ok=True)
        for filename in filenames:
            run_transformations(os.path.join(path, filename), out_dir_equivalent, transformations)

def run_transformations(image_path, out_dir, transformations, img_size=(64,64)):
    name, ext = os.path.splitext(os.path.basename(image_path))
    print(f'Creating {transformations} transformations of image {image_path} into {out_dir}...')

    image_data = cv2.imread(image_path)
    for i in range(transformations):
        transformation_filename = f'{name}-transformed_{i:04d}{ext}'
        transformation_path = os.path.join(out_dir, transformation_filename)
        transformed_data = transform_image_data(image_data)
        cv2.imwrite(transformation_path, transformed_data)

def transform_image_data(data):
    # Apply transformations to image data
    #TODO actually apply transformations
    return data

# endregion

# region Train Model

def train_model(data_dir, out_path):
    print(f'Training model from data in {data_dir} and exporting to {out_path}...')
    #TODO actually train model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)


# endregion

# region Main Execution

if str(args.generate) != '':
    generate_data(str(args.generate), str(args.directory), int(args.transformations))

if not args.skip_training:
    train_model(str(args.directory), str(args.output))

# endregion