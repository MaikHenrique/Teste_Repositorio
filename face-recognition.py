from recognition.face_recognition import train_generator, build_model
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import glob
from datetime import datetime
import tensorflow as tf
from utils.split_utils import read_splitted, load_images, load_image, read_data
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import csv


# from mpl_toolkits.mplot3d import Axes3D


def train():
    generator, test_generator = train_generator(data_dir, input_shape, 8, 8)
    model = build_model(input_shape)
    tf.keras.utils.plot_model(model, to_file='model.png')
    logdir = "classification_logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    model.fit_generator(generator, steps_per_epoch=40, epochs=20, validation_data=test_generator,
                        workers=1,
                        validation_steps=20,
                        callbacks=[
                            tensorboard_callback
                        ])
    model.save_weights("classifier.h5")
    return model


def test():
    model = build_model(input_shape)
    #model.load_weights("./bkp/classifier_xception.h5")
    model.load_weights(".classifier.h5")
    return model


def plot_distribution():
    X_train, _, y_train, _ = read_splitted(data_dir, test_size=.1)
    train_data = load_images(X_train, y_train, input_shape)
    train_embeddings = model.predict(np.array(list(train_data[:, 0])), batch_size=10)
    pca = PCA(n_components=2)
    components = pca.fit_transform(train_embeddings)
    # colors = {label: np.random.rand(3,) for label in y_train}
    colors = {'Indaia': [0.85865774, 0.37650632, 0.50745684], 'Mary': [0.20140247, 0.23166685, 0.5555025],
              'Bernado': [0.5573546, 0.09530458, 0.74360155], 'Carsten': [0.95897643, 0.40382865, 0.00107667],
              'Barbie': [0.50138109, 0.9929755, 0.93214165], 'Bauer': [0.66824179, 0.44940505, 0.86794612],
              'Shisha': [0.74865352, 0.10384365, 0.70145182], 'tedy': [0.33421823, 0.77872292, 0.29918022],
              'Bob': [0.03626006, 0.80469907, 0.52052402], 'Toby': [0.66475028, 0.38716534, 0.52905105],
              'Link': [0.21017388, 0.4747068, 0.559078], 'Ozzy': [0.39641091, 0.3549963, 0.97724448],
              'Selke': [0.95460547, 0.90028715, 0.62342923], 'Feijao': [0.72759571, 0.50212979, 0.3407233],
              'Ebel': [0.4662668, 0.18774943, 0.01507599], 'Edy': [0.45849246, 0.63398482, 0.49569488],
              'Sol': [0.32354957, 0.64919097, 0.14215046], 'Monica': [0.36595567, 0.69431044, 0.77532274]}
    labels = colors.keys()
    colors = [colors[label] for label in y_train]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Xception', fontsize=16)
    ax.scatter(components[:, 0], components[:, 1], c=colors, cmap="Set2_r", s=60)
    plt.show()
    fig.savefig('saved_figure_fig.png')
    ax.savefig('saved_figure_ax.png')


def export_tsv():
    X_train, _, y_train, _ = read_splitted(data_dir, test_size=.1)
    train_data = load_images(X_train, y_train, input_shape)
    train_embeddings = model.predict(np.array(list(train_data[:, 0])), batch_size=10)
    with open('embeddings.tsv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        for embedding in train_embeddings:
            writer.writerow(embedding)
        writeFile.close()
    with open('labels.tsv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        for label in y_train:
            writer.writerow([label])
        writeFile.close()


def test_open_data():
    X_train, _, y_train, _ = read_splitted(data_dir, test_size=.1)
    train_data = load_images(X_train, y_train, input_shape)
    train_embeddings = model.predict(np.array(list(train_data[:, 0])), batch_size=10)
    classifier = RadiusNeighborsClassifier(radius=.4, weights='distance', outlier_label='unknown')
    classifier.fit(train_embeddings, y_train)
    image, _ = load_image("test.jpg", 'x', input_shape)
    embeddings = model.predict(np.array([image]), batch_size=1)
    predicted = classifier.predict(embeddings)
    print(predicted)
    dist, ind = classifier.radius_neighbors(X=embeddings)
    print(dist[0])
    print(y_train[ind])


def evaluate_classifiers():
    X, y = read_data(data_dir)
    k_fold = StratifiedKFold(n_splits=10)
    folds = list(k_fold.split(X, y))
    for classifier_index in range(5):
        accuracy = []
        for i, (train_index, test_index) in enumerate(folds):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            train_data, test_data = load_images(X_train, y_train, input_shape), load_images(X_test, y_test, input_shape)

            trainX = model.predict(np.array(list(train_data[:, 0])), batch_size=10)
            trainy = y_train

            testX = model.predict(np.array(list(test_data[:, 0])), batch_size=10)
            testy = y_test

            if classifier_index == 0:
                classifier = RadiusNeighborsClassifier(radius=.4, weights='distance', outlier_label='unknown')
                print("Usando r-NN 0.4")
            elif classifier_index == 1:
                classifier = RadiusNeighborsClassifier(radius=.5, weights='distance', outlier_label='unknown')
                print("Usando r-NN 0.5")
            elif classifier_index == 2:
                classifier = RadiusNeighborsClassifier(radius=.7, weights='distance', outlier_label='unknown')
                print("Usando r-NN 0.7")
            elif classifier_index == 3:
                classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4, weights='distance')
                print("Usando k-NN k=5")
            else:
                classifier = LinearSVC()
                print("Usando SVM")
            classifier.fit(trainX, trainy)
            labels = [filename.split("\\")[-1] for filename in glob.glob(data_dir + "/*")]
            # labels.append('unknown')
            predict = classifier.predict(testX)
            acc = classifier.score(testX, testy)
            accuracy.append(acc)
            # print('Acurácia KNN: {}'.format(100 * acc))
            # report = classification_report(testy, predict)
            # print(report)

            # testMatrix = confusion_matrix(testy, predict)
            # sn.heatmap(testMatrix, annot=True, annot_kws={"size": "16"}, xticklabels=labels, yticklabels=labels)
            # plt.savefig("heatmap_{}.jpg".format(i))
            # plt.clf()

        print("{}: {}".format(classifier_index, accuracy))


if __name__ == '__main__':
    input_shape = (96, 96)

    # data_dir = "./recognition/augmented_data" 
    data_dir = "./recognition/augmented_data"
    #model = train()
    model = test()
    model.summary()

    # evaluate_classifiers()

    plot_distribution()

    # export_tsv()

    # test_open_data()
