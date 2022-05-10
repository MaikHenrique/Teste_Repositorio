import tensorflow as tf


class FaceDetetor:
    def __init__(self, grid_size, image_shape=(288, 288), coord=5.0, noobj=.5, lr=3e-4):
        self.grid_size = grid_size
        self.image_shape = image_shape
        self.coord = coord
        self.noobj = noobj
        self.lr = lr

    def create_model(self):
        shape = self.image_shape + (3,)
        i = tf.keras.layers.Input(shape=shape)

        # x = self.conv_layer((3, 3), 32, i)
        x = self.conv_layer((9, 9), 8, i)
        x = self.conv_layer((1, 1), 8, x)
        x = self.max_pooling(x)
        x = self.drop_out(x)

        # x = self.conv_layer((3, 3), 64, x)
        x = self.conv_layer((9, 9), 16, x)
        x = self.conv_layer((1, 1), 16, x)
        x = self.max_pooling(x)
        x = self.drop_out(x)

        # x = self.bottle_neck(128, 64, x)
        # x = self.max_pooling(x)
        # x = self.drop_out(x)
        #
        # x = self.bottle_neck(256, 128, x)
        # x = self.max_pooling(x)
        # x = self.drop_out(x)
        #
        # x = self.bottle_neck_x2(512, 256, x)
        # x = self.max_pooling(x)
        # x = self.drop_out(x)

        # X = self.conv_layer((3, 3), 1024, x)
        # X = self.conv_layer((3, 3), 1024, x)
        # x = self.bottle_neck_x2(1024, 512, x)
        # x = self.max_pooling(x)
        # x = self.drop_out(x)

        x = self.conv_layer((9, 9), 32, x)
        x = self.conv_layer((1, 1), 32, x)
        x = self.max_pooling(x)
        x = self.drop_out(x)

        x = self.conv_layer((9, 9), 64, x)
        x = self.conv_layer((1, 1), 64, x)
        x = self.max_pooling(x)
        x = self.drop_out(x)

        x = self.conv_layer((9, 9), 128, x)
        x = self.conv_layer((1, 1), 128, x)
        x = self.max_pooling(x)
        x = self.drop_out(x)

        cx2 = self.conv_layer((9, 9), 192, x)
        cx2 = self.drop_out(cx2)
        cx2 = self.conv_layer((1, 1), 192, cx2)
        cx2 = self.drop_out(cx2)

        cx3 = self.conv_layer((9, 9), 192, x)
        cx3 = self.drop_out(cx3)
        cx3 = self.conv_layer((1, 1), 192, cx3)
        cx3 = self.drop_out(cx3)

        cx4 = self.conv_layer((9, 9), 192, x)
        cx4 = self.drop_out(cx4)
        cx4 = self.conv_layer((1, 1), 192, cx4)
        cx4 = self.drop_out(cx4)

        prob = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation="sigmoid")(cx2)

        xy_center = tf.keras.layers.Conv2D(2, (1, 1), padding='same', activation="sigmoid")(cx3)

        # Não podemos usar sigmoid aqui pois o width pode ser maior que 1
        # utilizamos o ativação relu para evitar negativos
        wh = tf.keras.layers.Conv2D(2, (1, 1), padding='same', activation='relu')(cx4)

        x = tf.keras.layers.concatenate([prob, xy_center, wh])
        sequential = tf.keras.models.Model(inputs=i, outputs=x)
        adam = tf.keras.optimizers.Adam(learning_rate=self.lr)
        sequential.compile(optimizer=adam,
                           loss=self.loss,
                           metrics=[
                               self.iou,
                               # self.confidence
                           ]
                           )
        sequential.summary()
        return sequential

    def drop_out(self, x):
        return tf.keras.layers.Dropout(0.5)(x)

    def max_pooling(self, x):
        return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)

    def conv_layer(self, kernel, units, x):
        x = tf.keras.layers.Conv2D(units, kernel, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def bottle_neck(self, units, bottle_units, x):
        x = self.conv_layer((3, 3), units, x)
        x = self.conv_layer((1, 1), bottle_units, x)
        x = self.conv_layer((3, 3), units, x)
        return x

    def bottle_neck_x2(self, units, bottle_units, x):
        x = self.bottle_neck(units, bottle_units, x)
        x = self.conv_layer((1, 1), bottle_units, x)
        x = self.conv_layer((3, 3), units, x)
        return x

    def loss(self, gt, pred):
        return custom_loss(gt, pred, self.coord, self.noobj, self.grid_size)

    def iou(self, gt, pred):
        return metric_iou(gt, pred, self.grid_size)

    def confidence(self, gt, pred):
        return metric_conf(gt, pred, self.noobj, self.grid_size)


def calc_iou(gt_x_y, pred_x_y, gt_w_h, pred_w_h):
    pred_w_h = pred_w_h / 2
    gt_w_h = gt_w_h / 2
    pred_x_y_max = pred_x_y + pred_w_h
    gt_x_y_max = gt_x_y + gt_w_h
    pred_x_y_min = pred_x_y - pred_w_h
    gt_x_y_min = gt_x_y - gt_w_h
    reduce_max = tf.reduce_max([gt_x_y_min, pred_x_y_min], axis=0)
    reduce_min = tf.reduce_min([gt_x_y_max, pred_x_y_max], axis=0)
    xA = reduce_max[..., 0]
    yA = reduce_max[..., 1]
    xB = reduce_min[..., 0]
    yB = reduce_min[..., 1]

    interArea = tf.maximum(0.0, xB - xA + 1) * tf.maximum(0.0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    gt_area = gt_x_y_max - gt_x_y_min + 1
    gt_area = gt_area[..., 0] * gt_area[..., 1]
    pred_area = pred_x_y_max - pred_x_y_min + 1
    pred_area = pred_area[..., 0] * pred_area[..., 1]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (gt_area + pred_area - interArea)

    # return the intersection over union value
    return iou


def metric_conf(gt, pred, noobj=.5, grid_size=9):
    # gt_x_y = tf.reshape(gt[..., 1:3], [-1, 2])
    # pred_x_y = tf.reshape(pred[..., 1:3], [-1, 2])
    #
    # gt_w_h = tf.reshape(gt[..., 3:6], [-1, 2])
    # pred_w_h = tf.reshape(pred[..., 3:6], [-1, 2])

    has_face = tf.reshape(gt[..., 0], [-1])

    # iou = calc_iou(gt_x_y, pred_x_y, gt_w_h, pred_w_h) * has_face

    pred_confidence = tf.reshape(pred[..., 0], [-1])
    confidence_loss = pred_confidence
    confidence_loss = confidence_loss * has_face
    final_loss = tf.reduce_sum(tf.reshape(confidence_loss, [-1, grid_size * grid_size]), axis=1)
    return final_loss


def custom_loss(gt, pred, coord=5., noobj=.5, grid_size=9):
    gt_x_y = tf.reshape(gt[..., 1:3], [-1, 2])
    pred_x_y = tf.reshape(pred[..., 1:3], [-1, 2])
    first_loss = coord * tf.reduce_sum(tf.square(gt_x_y - pred_x_y), axis=1)

    gt_w_h = tf.reshape(gt[..., 3:6], [-1, 2])
    pred_w_h = tf.reshape(pred[..., 3:6], [-1, 2])
    second_loss = coord * tf.reduce_sum(tf.square(tf.sqrt(gt_w_h) - tf.sqrt(pred_w_h)), axis=1)
    bbox_loss = first_loss + second_loss

    has_face = tf.reshape(gt[..., 0], [-1])
    bbox_loss *= has_face

    iou = calc_iou(gt_x_y, pred_x_y, gt_w_h, pred_w_h) * has_face

    pred_confidence = tf.reshape(pred[..., 0], [-1])
    confidence_loss = (iou - pred_confidence) ** 2
    confidence_loss = confidence_loss * tf.maximum(noobj, has_face * coord)
    final_loss = bbox_loss #+ confidence_loss
    final_loss = tf.reduce_sum(tf.reshape(final_loss, [-1, grid_size * grid_size]), axis=1)
    return final_loss


def metric_iou(gt, pred, grid_size=9):
    gt_x_y = tf.reshape(gt[..., 1:3], [-1, 2])
    pred_x_y = tf.reshape(pred[..., 1:3], [-1, 2])
    gt_w_h = tf.reshape(gt[..., 3:6], [-1, 2])
    pred_w_h = tf.reshape(pred[..., 3:6], [-1, 2])
    has_face = tf.reshape(gt[..., 0], [-1])
    iou = calc_iou(gt_x_y, pred_x_y, gt_w_h, pred_w_h) * has_face
    iou = tf.reshape(iou, [-1, grid_size * grid_size])
    has_face = tf.reshape(has_face, [-1, grid_size * grid_size])
    iou = tf.reduce_sum(iou, axis=1)
    has_face = tf.reduce_sum(has_face, axis=1)
    return iou / has_face
