import tensorflow as tf
import numpy as np
import cv2
from detection.face_detector import metric_iou, custom_loss, metric_conf
from detection.utils import draw_boundingbox
import glob
import time

if __name__ == '__main__':
    tf.keras.utils.get_custom_objects().update({"loss": custom_loss, "iou": metric_iou, "confidence": metric_conf})
    image_shape = (288, 288)
    grid_size = 9
    # box_xmin = 0.407407
    # box_xmax = 0.674074
    # box_ymin = 0.187154
    # box_ymax = 0.420580
    # bbox = np.array([generate_grid_from_bbox([[box_xmin, box_ymin, box_xmax, box_ymax]], grid_size)], dtype='float32')

    model = tf.keras.models.load_model("teste.h5")
    files = glob.glob("detection_data/data/*.jpg")
    for file in files:
        img = cv2.imread(file)
        resized_image = cv2.resize(img, image_shape)
        array = np.array([resized_image], dtype=np.float)
        array *= 1 / 255
        start = time.time()
        test = model.predict(array, batch_size=1)
        draw_boundingbox(resized_image, test[0], grid_size, image_shape[0])
        print("Processado imagem em {} ms".format((time.time()-start)*1000))
        confidence = test[0][..., 0]
        cv2.imshow('test', resized_image)
        cv2.waitKey(1000)