import numpy as np
import cv2
import os

DIRNAME = os.path.dirname(__file__)
TRAIN_IMAGE_PATH = os.path.join(DIRNAME, "../data/images/vienna16.tif")
DEV_IMAGE_PATH = os.path.join(DIRNAME, "../data/images/vienna5.tif")
TEST_IMAGE_PATH = os.path.join(DIRNAME, "../data/images/vienna22.tif")
TRAIN_LABEL_PATH = os.path.join(DIRNAME, "../data/gt/vienna16.tif")
DEV_LABEL_PATH = os.path.join(DIRNAME, "../data/gt/vienna5.tif")
TEST_LABEL_PATH = os.path.join(DIRNAME, "../data/gt/vienna22.tif")
IMAGE_SIZE = 200


def load_dataset():
    train_image = cv2.imread(TRAIN_IMAGE_PATH)
    dev_image = cv2.imread(DEV_IMAGE_PATH)
    test_image = cv2.imread(TEST_IMAGE_PATH)
    train_label = cv2.imread(TRAIN_LABEL_PATH)
    dev_label = cv2.imread(DEV_LABEL_PATH)
    test_label = cv2.imread(TEST_LABEL_PATH)

    X_train = sample_image(train_image, IMAGE_SIZE)
    X_dev = sample_image(dev_image, IMAGE_SIZE)
    X_test = sample_image(test_image, IMAGE_SIZE)

    Y_train = sample_image(train_label, IMAGE_SIZE)
    Y_dev = sample_image(dev_label, IMAGE_SIZE)
    Y_test = sample_image(test_label, IMAGE_SIZE)

    Y_train[Y_train == 255] = 1
    Y_dev[Y_dev == 255] = 1
    Y_test[Y_test == 255] = 1
    print(Y_dev.shape)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def sample_image(image, image_size):
    samples = [image[row:row+image_size, col:col+image_size, :] for row in range(0, 5000 - image_size + 1, image_size) for col in range(0, 5000 - image_size + 1, image_size)]
    return np.stack(samples, axis=0)
