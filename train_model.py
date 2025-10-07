import argparse
import os
import shutil
import cv2
import albumentations as A
import numpy as np
import glob
import subprocess
import platform
from sklearn.preprocessing import LabelEncoder

# region Argument Parsing

parser = argparse.ArgumentParser(
    prog=f'python train_model.py',
    description='Training model for Magic Mirror prop used for Lockwood Immersive\'s HES Ball 2025 integration.',
    epilog='Created by Than | https://github.com/thana-than/LockwoodImmersiveMagicMirror')
parser.add_argument('-g', '--generate', default='training_data/raw', help='Generate extrapolated training data from folder.')
parser.add_argument('-d', '--directory', default='training_data/train', help='Directory of training data.')
parser.add_argument('-t', '--test-directory', default='training_data/test', help='Directory of testing data.')
parser.add_argument('-o', '--output', default='training_data/model.xml', help='Location to output model file.')
parser.add_argument('-s', '--skip-training', action='store_true', default=False, help='Skip model training step.')
args = parser.parse_args()

# endregion

# region Variables

CROP_TARGET_SIZE = 256
TARGET_SIZE_MULTIPLIER = 2
TRANSFORMATIONS = 1000
TEST_RATIO = 0.2

TARGET_RESIZE_MIN = CROP_TARGET_SIZE * TARGET_SIZE_MULTIPLIER

#endregion

# region Generate Data

#* Read https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/
seq = A.Compose([
    A.RandomCrop(height=CROP_TARGET_SIZE, width=CROP_TARGET_SIZE), 
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
        for i, filename in enumerate(filenames):
            print(f'\rTransforming images in folder {path}: {i+1} of {len(filenames)}', end="", flush=True)
            run_transformations(os.path.join(path, filename), train_dir_equivalent, test_dir_equivalent)
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

def run_transformations(image_path, train_dir, test_dir, transformations = TRANSFORMATIONS):
    if not image_path.lower().endswith(('.jpg', '.png')):
        return

    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    
    height, width = image_data.shape[:2]
    if height < TARGET_RESIZE_MIN or width < TARGET_RESIZE_MIN:
        print(f'\nWarning: Image {image_path} is smaller than target size {TARGET_RESIZE_MIN}x{TARGET_RESIZE_MIN}. Resizing.')
        image_data = cv2.resize(image_data, (max(width, TARGET_RESIZE_MIN), max(height, TARGET_RESIZE_MIN)))

    name, ext = os.path.splitext(os.path.basename(image_path))
    #* Apply transformations to image data
    for i in range(transformations):
        out_dir = test_dir if i < transformations * TEST_RATIO else train_dir
        augmented = seq(image=image_data)['image']
        transformation_filename = f'{name}-transformed_{i:04d}{ext}'
        transformation_path = os.path.join(out_dir, transformation_filename)
        
        #augmented = (augmented * 255).clip(0, 255).astype('uint8')
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        cv2.imwrite(transformation_path, augmented)

# endregion

# region Train Model

def train_model(train_dir, test_dir, out_path):
    print(f'TRAIN MODEL STEP')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    train_features, train_labels = extract_hog_features(train_dir)
    print(f'Training model from data in {train_dir} and exporting to {out_path}...')
    svm = train_svm(train_features, train_labels)
    svm.save(out_path)

    print(f'TESTING STEP')
    test_features, test_labels = extract_hog_features(test_dir)
    print(f'Evaluating SVM model with test data from {test_dir}...')
    accuracy = evaluate_svm(svm, test_features, test_labels)
    print(f'Test accuracy: {accuracy:.2f}')

def extract_hog_features(image_folder, win_size=(CROP_TARGET_SIZE, CROP_TARGET_SIZE), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=9):
    print(f'Extracting HOG features from {image_folder}...')
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
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
            features.append(feature)
            labels.append(category)
        print()

    features = np.array(features).squeeze()
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
def train_svm(train_features, train_labels, kernel=cv2.ml.SVM_LINEAR):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(kernel)
    train_data = cv2.ml.TrainData_create(train_features, cv2.ml.ROW_SAMPLE, train_labels)
    svm.train(train_data)

    return svm

def evaluate_svm(svm, test_features, test_labels):
    predicted_labels = svm.predict(test_features)[1].ravel()
    accuracy = np.mean(predicted_labels == test_labels)
    return accuracy

# endregion

# region Main Execution

if str(args.generate) != '':
    generate_data(str(args.generate), str(args.directory), str(args.test_directory))

if not args.skip_training:
    train_model(str(args.directory), str(args.test_directory), str(args.output))

print("Done.")

# endregion