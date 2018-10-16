import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
from keras import backend as K

DIRNAME = os.path.dirname(__file__)
TRAIN_IMAGE_PATH = os.path.join(DIRNAME, "../data/images/vienna16.tif")
DEV_IMAGE_PATH = os.path.join(DIRNAME, "../data/images/vienna5.tif")
TEST_IMAGE_PATH = os.path.join(DIRNAME, "../data/images/vienna22.tif")
TRAIN_LABEL_PATH = os.path.join(DIRNAME, "../data/gt/vienna16.tif")
DEV_LABEL_PATH = os.path.join(DIRNAME, "../data/gt/vienna5.tif")
TEST_LABEL_PATH = os.path.join(DIRNAME, "../data/gt/vienna22.tif")
IMAGE_SIZE = 200


def load_dataset(image_size=IMAGE_SIZE):
    train_image = cv2.imread(TRAIN_IMAGE_PATH)
    dev_image = cv2.imread(DEV_IMAGE_PATH)
    test_image = cv2.imread(TEST_IMAGE_PATH)
    train_label = cv2.imread(TRAIN_LABEL_PATH, 0).reshape(5000, 5000, 1)
    dev_label = cv2.imread(DEV_LABEL_PATH, 0).reshape(5000, 5000, 1)
    test_label = cv2.imread(TEST_LABEL_PATH, 0).reshape(5000, 5000, 1)

    X_train = sample_image(train_image, IMAGE_SIZE)
    X_dev = sample_image(dev_image, IMAGE_SIZE)
    X_test = sample_image(test_image, IMAGE_SIZE)

    Y_train = sample_image(train_label, IMAGE_SIZE)
    Y_dev = sample_image(dev_label, IMAGE_SIZE)
    Y_test = sample_image(test_label, IMAGE_SIZE)

    Y_train[Y_train == 255] = 1
    Y_dev[Y_dev == 255] = 1
    Y_test[Y_test == 255] = 1

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def sample_image(image, image_size):
    samples = [image[row:row+image_size, col:col+image_size, :] for row in range(0, 5000 - image_size + 1, image_size) for col in range(0, 5000 - image_size + 1, image_size)]
    return np.stack(samples, axis=0)

def show_dataset():
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_dataset()
    images = [random.randint(0, 5000*5000/IMAGE_SIZE/IMAGE_SIZE - 1) for _ in range(20)]

    show_grid(X_train[images], Y_train[images])
    show_grid(X_dev[images], Y_dev[images])
    show_grid(X_test[images], Y_test[images])


def show_grid(train, label):
    fig = plt.figure(figsize=(100, 100))
    columns = 4
    rows = 5
    for i in range(0, rows):
        fig.add_subplot(rows, columns, 4 * i + 1)
        plt.imshow(train[2*i])
        fig.add_subplot(rows, columns, 4 * i + 2)
        plt.imshow(label[2*i].squeeze() * 255, cmap="gray")
        fig.add_subplot(rows, columns, 4 * i + 3)
        plt.imshow(train[2*i + 1])
        fig.add_subplot(rows, columns, 4 * i + 4)
        plt.imshow(label[2*i + 1].squeeze() * 255, cmap="gray")
    plt.show()


def baseline():
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_dataset()
    print(Y_train.sum()/5000/5000)
    print(Y_dev.sum()/5000/5000)
    print(Y_test.sum()/5000/5000)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
