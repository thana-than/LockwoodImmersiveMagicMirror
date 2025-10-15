import argparse
import os
import shutil
import cv2
import albumentations as A
import numpy as np
import glob
import subprocess
import platform
import json
from sklearn.preprocessing import LabelEncoder

# region Argument Parsing

parser = argparse.ArgumentParser(
    prog=f'python train_model.py',
    description='Training model for Magic Mirror prop used for Lockwood Immersive\'s HES Ball 2025 integration.',
    epilog='Created by Than | https://github.com/thana-than/LockwoodImmersiveMagicMirror')
parser.add_argument('-g', '--generate', default='training_data/raw', help='Generate extrapolated training data from folder.')
parser.add_argument('-d', '--directory', default='training_data/train', help='Directory of training data.')
parser.add_argument('-t', '--test-directory', default='training_data/test', help='Directory of testing data.')
parser.add_argument('-m', '--model-directory', default='model', help='Location of model directory. Includes model, HOG descriptor, and features.')
parser.add_argument('-s', '--skip-training', action='store_true', default=False, help='Skip model training step.')
parser.add_argument('--regenerate-hog', action='store_true', default=False, help='Regenerate HOG descriptor and features.')
args = parser.parse_args()

# endregion

# region JSON Configuration
P_CROP_SIZE = 'crop_size'
P_MIN_SIZE = 'min_size'
P_TRANSFORMATIONS = 'transformations'
P_TEST_RATIO = 'test_ratio'

P_HOG_BLOCK_SIZE = 'hog_block_size'
P_HOG_BLOCK_STRIDE = 'hog_block_stride'
P_HOG_CELL_SIZE = 'hog_cell_size'
P_HOG_NBINS = 'hog_nbins'

GENERATE_IMAGES = str(args.generate) != ''

MODEL_DIR = str(args.model_directory)
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.xml')
HOG_DESCRIPTOR_PATH = os.path.join(MODEL_DIR, 'hog_descriptor.xml')
def get_features_path(feature_type):
    return os.path.join(MODEL_DIR, f'{feature_type}_features.npz')


DEFAULT_ONLY_PROPERTIES = [P_CROP_SIZE, P_MIN_SIZE, P_HOG_BLOCK_SIZE, P_HOG_BLOCK_STRIDE, P_HOG_CELL_SIZE, P_HOG_NBINS]

DEFAULT_JSON_DATA = {
    "default": {
        P_CROP_SIZE: 128,
        P_MIN_SIZE: 256,
        P_HOG_BLOCK_SIZE: [16, 16],
        P_HOG_BLOCK_STRIDE: [8, 8],
        P_HOG_CELL_SIZE: [8, 8],
        P_HOG_NBINS: 9,
        P_TRANSFORMATIONS: 1000,
        P_TEST_RATIO: 0.2,

    },
    "none": {
        P_TRANSFORMATIONS: 10
    }
}

json_file = 'train_model.json'


def write_default_json(path):
    with open(path, 'w') as f:
        json.dump(DEFAULT_JSON_DATA, f, indent=4)
    return DEFAULT_JSON_DATA

def read_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = DEFAULT_JSON_DATA
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
    
    return data

def get_channel_property(channel, property_name):
    #* Try to get property from JSON data, falling back to defaults if not found
    if channel in json_data:
        if property_name in json_data[channel]:
            if property_name not in DEFAULT_ONLY_PROPERTIES:
                    return json_data[channel][property_name]
            else:
                print(f'Warning: Property {property_name} is only allowed in default section. Using default value.')
            
    if "default" in json_data:
        if property_name in json_data["default"]:
            return json_data["default"][property_name]
        
    if channel in DEFAULT_JSON_DATA:
        if property_name in DEFAULT_JSON_DATA[channel]:
            if property_name not in DEFAULT_ONLY_PROPERTIES:
                return DEFAULT_JSON_DATA[channel][property_name]
        
    if "default" in DEFAULT_JSON_DATA:
        if property_name in DEFAULT_JSON_DATA["default"]:
            return DEFAULT_JSON_DATA["default"][property_name]
        
    raise ValueError(f'Property {property_name} not found for channel {channel} or default.')
    
# endregion

# region Generate Data

#* Read https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/
def get_augmentation_pipeline(crop_target_size):
    return A.Compose([
    A.RandomCrop(height=crop_target_size, width=crop_target_size), 
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8),
        p=1.0
    ),
    A.OneOf([
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25), p=1.0),
        A.GridDropout(ratio=0.5, unit_size_range=(5, 10), p=1.0)
    ], p=0.5),
    A.OneOf([
        A.ToGray(p=0.3),
        A.ChannelDropout(channel_drop_range=(1, 1), p=0.3),
    ], p=0.2),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
        A.RandomGamma(gamma_limit=(80, 120), p=0.8),
    ], p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=1.0),
    # A.GaussNoise(p=0.5),
    #A.Normalize(),
])

def generate_data(in_dir, train_dir, test_dir):
    print("GENERATE DATA STEP")
    print(f'Generating extrapolated data from {in_dir}...')

    #* Clear output directories
    recreate_dir(train_dir)
    recreate_dir(test_dir)

    for path, subdirs, filenames in os.walk(in_dir):
        train_dir_equivalent = os.path.join(train_dir, os.path.relpath(path, in_dir))
        test_dir_equivalent = os.path.join(test_dir, os.path.relpath(path, in_dir))
        os.makedirs(train_dir_equivalent, exist_ok=True)
        os.makedirs(test_dir_equivalent, exist_ok=True)

        #* Determine channel properties from JSON data
        channel = os.path.basename(path)
        min_size = get_channel_property(channel, P_MIN_SIZE)
        test_ratio = get_channel_property(channel, P_TEST_RATIO)
        transformations = get_channel_property(channel, P_TRANSFORMATIONS)
        crop_size = get_channel_property(channel, P_CROP_SIZE)
        augmentation_pipeline = get_augmentation_pipeline(crop_size)

        for i, filename in enumerate(filenames):
            print(f'\rTransforming images in folder {path}: {i+1} of {len(filenames)}', end="", flush=True)
            run_transformations(os.path.join(path, filename), train_dir_equivalent, test_dir_equivalent, min_size, test_ratio, transformations, augmentation_pipeline)
        print()

def recreate_dir(dir_path):
    print(f'Recreating {dir_path}...')
    if os.path.exists(dir_path):
        system = platform.system()

        #* Attempt to delete directories through native commands for speed
        try:
            if system == "Windows":
                #* Read https://stackoverflow.com/questions/186737/whats-the-fastest-way-to-delete-a-large-folder-in-windows
                subprocess.run(["del", "/f", "/s", "/q", dir_path],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["rmdir", "/s", "/q", dir_path],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["rm", "-rf", dir_path],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            #* Fallback to pure Python if something fails
            shutil.rmtree(dir_path, ignore_errors=True)

    os.makedirs(dir_path)

def run_transformations(image_path, train_dir, test_dir, min_size, test_ratio, transformations, augmentation_pipeline):
    if not image_path.lower().endswith(('.jpg', '.png')):
        return

    image_data = cv2.imread(image_path)
    if image_data is None:
        print(f'\nWarning: Unable to read image {image_path}. Skipping.')
        return
    
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    
    height, width = image_data.shape[:2]
    if height < min_size or width < min_size:
        print(f'\nWarning: Image {image_path} is smaller than target size {min_size}x{min_size}. Resizing.')
        image_data = cv2.resize(image_data, (max(width, min_size), max(height, min_size)))

    name, ext = os.path.splitext(os.path.basename(image_path))
    #* Apply transformations to image data
    for i in range(transformations):
        out_dir = test_dir if i < transformations * test_ratio else train_dir
        augmented = augmentation_pipeline(image=image_data)['image']
        transformation_filename = f'{name}-transformed_{i:04d}{ext}'
        transformation_path = os.path.join(out_dir, transformation_filename)
        
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        cv2.imwrite(transformation_path, augmented)

# endregion

# region Train Model

def train_model(train_dir, test_dir, out_path):
    hog, fresh= get_hog_descriptor()
    
    print(f'TRAIN MODEL STEP')
    train_features, train_labels = get_hog_features(hog, train_dir, fresh)
    print(f'Training model from data in {train_dir}...')
    svm = train_svm(train_features, train_labels)
    print(f'Saving svm model to {out_path}...')
    svm.save(out_path)

    print(f'TESTING STEP')
    test_features, test_labels = get_hog_features(hog, test_dir, fresh)
    print(f'Evaluating SVM model with test data from {test_dir}...')
    accuracy = evaluate_svm(svm, test_features, test_labels)
    print(f'Test accuracy: {accuracy:.2f}')

def get_hog_descriptor():
    if GENERATE_IMAGES or args.regenerate_hog or not os.path.exists(HOG_DESCRIPTOR_PATH):
        hog = build_hog_descriptor()
        print(f'Saving HOG Descriptor to {HOG_DESCRIPTOR_PATH}...')
        hog.save(HOG_DESCRIPTOR_PATH)
        fresh = True
    else:
        print(f'Loading HOG Descriptor from {HOG_DESCRIPTOR_PATH}...')
        hog = cv2.HOGDescriptor(HOG_DESCRIPTOR_PATH)
        fresh = False
    return hog, fresh

def build_hog_descriptor():
    crop_size = get_channel_property('default', P_CROP_SIZE)
    block_size=get_channel_property('default', P_HOG_BLOCK_SIZE)
    block_stride=get_channel_property('default', P_HOG_BLOCK_STRIDE)
    cell_size=get_channel_property('default', P_HOG_CELL_SIZE)
    nbins=get_channel_property('default', P_HOG_NBINS)

    hog = cv2.HOGDescriptor((crop_size, crop_size), block_size, block_stride, cell_size, nbins)
    print(f'Built new HOG Descriptor with win size {crop_size}, block size {block_size}, block stride {block_stride}, cell size {cell_size}, and {nbins} bins.')

    return hog

def get_hog_features(hog, image_folder, require_regen=False):
    features_type = os.path.basename(image_folder)
    feature_path = get_features_path(features_type)
    print(f'Getting HOG features for {features_type} data...')
    if require_regen or not os.path.exists(feature_path):
        features, labels = extract_hog_features(hog, image_folder)
        print(f'Saving {features_type} features to {feature_path}...')
        np.savez_compressed(feature_path, features=features, labels=labels)
    else:
        print(f'Loading {features_type} features and labels from {feature_path}...')
        data = np.load(feature_path)
        features = data['features']
        labels = data['labels']

    return features, labels

def extract_hog_features(hog, image_folder):
    print(f'Extracting HOG features from {image_folder}...')
    features = []
    labels = []

    for category_folder in glob.glob(os.path.join(image_folder, '*')):
        category = os.path.basename(category_folder)
        image_files = glob.glob(os.path.join(category_folder, '*.jpg')) + glob.glob(os.path.join(category_folder, '*.png'))
        image_files_length = len(image_files)
        for i, image_path in enumerate(image_files):
            print(f'\rExtracting HOG features from category {category}: {i+1} of {image_files_length}', end="", flush=True)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            feature = hog.compute(image)
            features.append(feature.flatten())
            labels.append(category)
        print()

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)

    #* See https://stackoverflow.com/questions/78767468/python-errors-when-calling-cv2-ml-traindata-create-responses-data-type
    labels_unique = np.unique(labels)
    labels_int = labels.copy()
    i=0
    for unique_label in labels_unique:
        print(str(i)+': '+unique_label)
        labels_int[labels == unique_label] = i
        i += 1
    labels_int = labels_int.astype(np.int32)

    return features, labels_int

#* Read https://www.opencvhelp.org/tutorials/deep-learning/training-models/
#* And https://machinelearningmastery.com/opencv_object_detection/
def train_svm(train_features, train_labels):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(1.0)
    svm.setGamma(0.1)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6))
    svm.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)

    return svm

def evaluate_svm(svm, test_features, test_labels):
    predicted_labels = svm.predict(test_features)[1].ravel()
    accuracy = np.mean(predicted_labels == test_labels)
    return accuracy

# endregion

# region Main Execution

json_data = read_json(json_file)

if GENERATE_IMAGES:
    generate_data(str(args.generate), str(args.directory), str(args.test_directory))

if not args.skip_training:
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_model(str(args.directory), str(args.test_directory), MODEL_PATH)

print("Done.")

# endregion