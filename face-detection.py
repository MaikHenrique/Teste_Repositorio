from detection.face_detector import FaceDetetor, metric_iou, custom_loss
from detection.utils import draw_boundingbox, generate_grid_from_bbox
from detection.face_generator import FaceGenerator, split_dataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from datetime import datetime


def lr_schedule(lr):
    start_epoch = 0

    def func(epoch):
        drop = 0.5
        epochs_drop = 25.0
        lrate = lr
        if epoch > start_epoch:
            lrate = lr * math.pow(drop,
                                  math.floor((1 + epoch) / epochs_drop))
        return lrate

    return tf.keras.callbacks.LearningRateScheduler(func, verbose=1)


if __name__ == '__main__':
    image_shape = (288, 288)
    batch_size = 16
    grid_size = 9
    initial_lr = 5e-3
    model = FaceDetetor(grid_size, image_shape, lr=initial_lr).create_model()
    train_generator, test_generator = split_dataset(target_size=image_shape, batch_size=batch_size,
                                                    labels_path="detection_data_augmented/labels",
                                                    images_path="detection_data_augmented/data",
                                                    test_size=.3
                                                    )
    # train_generator, test_generator = split_dataset(target_size=image_shape, batch_size=batch_size,
    #                                                 labels_path="C:\\Bugios\\wider_face_split\\wider_face_val_bbx_gt.txt",
    #                                                 images_path="C:\\Bugios\\WIDER_val\\images\\",
    #                                                 type="wider"
    #                                                 )
    logdir = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    history = model.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator), epochs=100,
                                  validation_data=test_generator,
                                  validation_steps=len(test_generator), workers=8,
                                  callbacks=[
                                      # lr_schedule(initial_lr),
                                      tensorboard_callback
                                  ]
                                  )
    box_xmin = 0.407407
    box_xmax = 0.674074
    box_ymin = 0.187154
    box_ymax = 0.420580
    bbox = np.array([generate_grid_from_bbox([[box_xmin, box_ymin, box_xmax, box_ymax]], grid_size)], dtype='float32')
    img = cv2.imread("detection_data/data/111.jpg")
    # img = cv2.imread("C:\\Bugios\\WIDER_val\\images\\0--Parade\\0_Parade_marchingband_1_20.jpg")
    resized_image = cv2.resize(img, image_shape)
    array = np.array([resized_image], dtype=np.float)
    array *= 1 / 255
    #estava cometado daqui
    history = model.fit(array, bbox, batch_size=1, epochs=120, validation_data=(array, bbox))
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(["loss", "val_loss"])
    plt.title("Perda (Menor melhor)")
    plt.show(block=True)
    #
    plt.plot(history.history['iou'])
    plt.plot(history.history['val_iou'])
    plt.plot(history.history['confidence'])
    plt.plot(history.history['val_confidence'])
    plt.legend(["iou", "val_iou", "confidence", "val_confidence"])
    plt.title("Métricas IoU e Confidence (Maior melhor)")
    plt.show(block=True)
    #ate aqui

    test = model.predict(array, batch_size=1)
    print(test[0][2][4])
    draw_boundingbox(resized_image, test[0], grid_size, image_shape[0])
    confidence = test[0][..., 0]
    cv2.imshow('test', resized_image)
    cv2.waitKey(0)
    flatten = np.array(test).reshape((-1))

    print(metric_iou(bbox, test, grid_size))

    model.save("teste.h5")
    # f = open("model.json", "w")
    # f.write(model.to_json())
    # f.close()
    #
    # model.save_weights("model_weights.h5")
