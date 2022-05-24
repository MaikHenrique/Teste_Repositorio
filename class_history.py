from detection.face_detector import FaceDetetor, metric_iou, custom_loss
from detection.utils import draw_boundingbox, generate_grid_from_bbox
from detection.face_generator import FaceGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_shape = (288, 288)
batch_size = 16
grid_size = 9

#outro bloco
# Adicionar data augmentation e dataset para validação (Atualmente estamos utilizando o mesmo para treinamento e validação)
model = FaceDetetor(grid_size, image_shape).create_model()
generator = FaceGenerator(target_size=image_shape, batch_size=batch_size)
history = model.fit_generator(generator=generator, steps_per_epoch=len(generator), epochs=5,
                              validation_data=generator,
                              validation_steps=1, workers=8)

#outro bloco
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["loss", "val_loss"])
plt.title("Perda (Menor melhor)")
plt.show(block=True)

#outro bloco
plt.plot(history.history['iou'])
plt.plot(history.history['val_iou'])
plt.legend(["iou", "val_iou"])
plt.title("IoU - Intersection over Union (Maior melhor)")
plt.show(block=True)
