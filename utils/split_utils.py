import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
from itertools import groupby
from PIL import Image


def read_splitted(data_dir, test_size=.15):
    X, y = read_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def read_data(data_dir):
    files = glob.glob(data_dir + '/*/*.[jp][pn][g]')
    data = [[file, file.split(os.sep)[-2]] for file in files]
    data = np.array(data)
    return data[..., 0], data[..., 1]


def group_by_class(X):
    sorted_x = sorted(X, key=lambda file: file.split(os.sep)[-2])
    return {k: list(v) for k, v in groupby(sorted_x, key=lambda file: file.split(os.sep)[-2])}


def load_image(filepath, label, input_shape):
    image = Image.open(filepath)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(input_shape)
    image = np.array(image) / 255
    return [image, label]


def load_images(X, y, input_shape):
    return np.array([load_image(filepath, label, input_shape) for filepath, label in zip(X, y)])


if __name__ == '__main__':
    splitted = read_splitted("../recognition/data")
    print(group_by_class(splitted[0]))
