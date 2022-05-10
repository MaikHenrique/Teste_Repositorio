import cv2
import math
import numpy as np


def draw_boundingbox(img, output, grid_size, image_width):
    cell_proportional_size = 1 / grid_size
    color = (0, 255, 0)
    np.argmax(output[..., 0])
    for row in range(grid_size):
        for column in range(grid_size):
            cell = output[row][column]
            if cell[0] > 0.30:
                print(cell)
                x = cell[1]
                y = cell[2]
                w = cell[3]
                h = cell[4]

                # Os valores de x, y, w, h estão relativos ao tamanho da celula,
                # aqui deixamos eles relativos ao tamanho da imagem
                x = x * cell_proportional_size + column * cell_proportional_size
                y = y * cell_proportional_size + row * cell_proportional_size
                w = w * cell_proportional_size
                h = h * cell_proportional_size

                # Calculamos os dois pontos para a bounding box
                xmin = x - w / 2
                xmax = x + w / 2
                ymin = y - h / 2
                ymax = y + h / 2
                # print("xmin, xmax, ymin, ymax: {}, {}, {}, {}".format(xmin, xmax, ymin, ymax))
                draw_bbox(img, xmin, xmax, ymin, ymax, image_width, color)


def draw_bbox(img, box_xmin, box_xmax, box_ymin, box_ymax, image_width, color=(0, 255, 0)):
    xmin, xmax = int(box_xmin * image_width), int(box_xmax * image_width)
    ymin, ymax = int(box_ymin * image_width), int(box_ymax * image_width)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color)


def generate_grid_from_bbox(bboxes, grid_size):
    # Calcula em qual celula está o ponto central.
    # Proporcional a imagem
    cell_proportional_size = 1 / grid_size
    network_output = np.zeros((grid_size, grid_size, 5))
    for bbox in bboxes:
        box_xmin, box_ymin, box_xmax, box_ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        x_center = (box_xmin + box_xmax) / 2
        y_center = (box_ymin + box_ymax) / 2
        grid_column = int(x_center // cell_proportional_size)
        grid_row = int(y_center // cell_proportional_size)
        x_remainder = x_center % cell_proportional_size
        y_remainder = y_center % cell_proportional_size
        y_center_cell = y_remainder / cell_proportional_size
        x_center_cell = x_remainder / cell_proportional_size
        w = math.fabs(box_xmax - box_xmin) / cell_proportional_size
        h = math.fabs(box_ymax - box_ymin) / cell_proportional_size
        # print("Grid atual(Começando de zero): {}, {}".format(grid_row, grid_column))
        # print("Ponto central: {}, {}".format(y_center, x_center))
        # print("Ponto central na célula: {}, {}".format(y_center_cell, x_center_cell))
        # print("Altura e largura proporcional a ceula: {}, {}".format(h, w))
        network_output[grid_row][grid_column] = [1, x_center_cell, y_center_cell, w, h]
    return network_output
