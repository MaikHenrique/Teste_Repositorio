import glob

import cv2
import math
import numpy as np
from tensorflow import keras

import detection.utils as utils
from sklearn.model_selection import train_test_split


class FaceGenerator(keras.utils.Sequence):
    def __init__(self, target_size, x, y,
                 batch_size=16, grid_size=9, data_type='yolo'):
        self.batch_size = batch_size
        self.target_size = target_size
        self.grid_size = grid_size
        self.x = x
        self.y = y
        # if data_type == 'yolo':
        #     self.x = glob.glob(images_path + "/*.jpg")
        #     self.y = self.read_labels(glob.glob(labels_path + "/*.txt"))
        # else:
        #     self.x, self.y = self.read_wider(images_path, labels_path)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = np.array([cv2.resize(cv2.imread(file_name), self.target_size) for file_name in batch_x], dtype=np.float)
        x *= 1 / 255  # Normalizamos os pixels para ficar entre 0 e 1
        return x, np.array(batch_y)

    def read_labels(self, y):
        new_labels = []
        for label_file in y:
            with open(label_file, 'r') as lb_file:
                bboxes = []
                for label in lb_file.readlines():
                    bbox = np.asarray(label.split(" ")[1:]).astype(np.float)
                    bboxes.append(bbox)
                new_labels.append(utils.generate_grid_from_bbox(bboxes, self.grid_size))
        return new_labels


def split_dataset(target_size, images_path='detection_data/data', labels_path='detection_data/labels', batch_size=16,
                  grid_size=9, type="yolo", test_size=.3):
    if type == "yolo":
        x_files = glob.glob(images_path + "/*.jpg")
        y_files = read_labels(glob.glob(labels_path + "/*.txt"), grid_size)
    else:
        x_files, y_files = read_wider(images_path, labels_path, grid_size)
    x_train, x_test, y_train, y_test = train_test_split(x_files, y_files, test_size=test_size)
    train_generator = FaceGenerator(target_size, x=x_train, y=y_train, batch_size=batch_size, grid_size=grid_size)
    test_generator = FaceGenerator(target_size, x=x_test, y=y_test, batch_size=batch_size, grid_size=grid_size)
    return train_generator, test_generator


# Rotina para ler arquivos no formato do dataset wider
def read_wider(images_path, labels_path, grid_size):
    images = []
    new_labels = []
    with open(labels_path, 'r') as lb_file:
        file_name = lb_file.readline()
        while file_name:
            file_path = images_path + "/" + file_name
            file_path = file_path.replace("\n", "")
            height, width, __ = cv2.imread(file_path).shape
            images.append(file_path)
            faces_count = int(lb_file.readline())
            image_bboxes = []
            for i in range(faces_count):
                bbox_params = lb_file.readline().split(" ")
                if not bbox_params[7] != '0':
                    x, y, w, h = np.array(bbox_params[:4], dtype=np.float)
                    x /= width
                    y /= height
                    x_max = (x + w) / width
                    y_max = (y + h) / height
                    bbox = [x, y, x_max, y_max]
                    image_bboxes.append(bbox)
            new_labels.append(utils.generate_grid_from_bbox(image_bboxes, grid_size))
            file_name = lb_file.readline()
    return images, new_labels



def read_labels(files, grid_size):
    new_labels = []
    for label_file in files:
        with open(label_file, 'r') as lb_file:
            bboxes = []
            for label in lb_file.readlines():
                bbox = np.asarray(label.split(" ")[1:]).astype(np.float)
                bboxes.append(bbox)
            new_labels.append(utils.generate_grid_from_bbox(bboxes, grid_size))
    return new_labels
