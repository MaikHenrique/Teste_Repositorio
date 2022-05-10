from __future__ import print_function
import tensorflow.keras.backend as K
import tensorflow as tf
import glob
from PIL import Image
import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from recognition.triplet import TripletSemiHardLoss
from utils.split_utils import read_splitted, group_by_class, load_image

semi_hard_loss = TripletSemiHardLoss(margin=.2)


def build_generators(data_dir, image_shape, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        validation_split=0.3)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False,
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False,
        subset='validation')
    return train_generator, validation_generator


def triplet_loss(labels, embeddings):
    labels = K.reshape(labels, shape=[-1])

    return semi_hard_loss.call(labels, embeddings)


def train_generator(data_dir, input_shape, num_images_per_class, num_classes_per_batch, test_size=.15):
    X_train, X_test, y_train, y_test = read_splitted(data_dir, test_size=test_size)
    paths_train = group_by_class(X_train)
    paths_test = group_by_class(X_test)

    train_generator = get_generator(paths_train, input_shape, num_images_per_class, num_classes_per_batch)
    test_generator = get_generator(paths_test, input_shape, num_images_per_class, num_classes_per_batch)
    return train_generator, test_generator


def get_generator(paths, input_shape, num_images_per_class, num_classes_per_batch):
    def generator():
        while True:
            # Sample the labels that will compose the batch
            labels = np.random.choice(list(paths.keys()),
                                      num_classes_per_batch,
                                      replace=False)
            for label in labels:
                images = paths.get(label)
                for _ in range(num_images_per_class):
                    choice = np.random.choice(range(len(images)), 1)
                    filepath = images[choice[0]]
                    yield load_image(filepath, label, input_shape)

    def batch_generator():
        while 1:
            g = generator()
            batch_size = num_classes_per_batch * num_images_per_class
            batch_x = np.zeros((batch_size,) + (input_shape[0], input_shape[1], 3))
            batch_y = np.empty([batch_size], dtype="S20")
            for i in range(batch_size):
                generated = next(g)
                batch_y[i] = generated[1]
                batch_x[i] = generated[0]
            yield batch_x, batch_y

    return batch_generator()


def mean_norm(_, embeddings):
    return tf.reduce_mean(tf.norm(embeddings, axis=1))


def svm_accuracy(labels, embeddings):
    def accuracy_func(x, y):
        y = list(y.numpy())
        svm = LinearSVC()
        svm.fit(x, y)
        return svm.score(x, y)

    return tf.py_function(accuracy_func, inp=[embeddings, K.reshape(labels, shape=[-1])], Tout=tf.float32,
                          name="svm_accuracy")


def knn_accuracy(labels, embeddings):
    def accuracy_func(x, y):
        y = list(y.numpy())
        knn = RadiusNeighborsClassifier(radius=.5, outlier_label='unknown',
                                   weights='distance')
        knn.fit(x, y)
        return knn.score(x, y)

    return tf.py_function(accuracy_func, inp=[embeddings, K.reshape(labels, shape=[-1])], Tout=tf.float32,
                          name="knn_accuracy")


def build_model(image_shape):
    input_shape = image_shape[0], image_shape[1], 3

    # initial_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, input_shape=input_shape)
    # initial_model = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape)
    initial_model = tf.keras.applications.xception.Xception(include_top=False, input_shape=input_shape)
    last_layer = initial_model.output
    x = tf.keras.layers.Flatten()(last_layer)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Lambda(lambda embeddings: K.l2_normalize(embeddings, axis=1))(x)
    model = tf.keras.models.Model(initial_model.input, x)
    # for layer in model.layers[:18]:
    #     layer.trainable = False

    adam = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)
    model.compile(loss=triplet_loss, optimizer=adam, metrics=[
        # knn_accuracy, svm_accuracy
    ])
    return model
