import imgaug as ia
import shutil
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import numpy as np
import glob
import os

file_sequence = 1

seq = iaa.Sequential([
        # iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order


def repeat(item, times):
    return [item for _ in range(times)]


def augment_image(image, bounding_box):
    shape = image.shape

    image_height, image_width, _ = shape
    not_augmented_bboxes = []
    for bbox in bounding_box:
        x1 = bbox[0] * image_width
        x2 = bbox[2] * image_width
        y1 = bbox[1] * image_height
        y2 = bbox[3] * image_height
        not_augmented_bboxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

    bbs = BoundingBoxesOnImage(not_augmented_bboxes, shape=shape)
    images = repeat(image, 9)
    bboxes = repeat(bbs, 9)
    images_aug, bbss_aug = seq(images=images, bounding_boxes=bboxes)
    images_aug.append(image)
    bbss_aug.append(bbs)

    all_images_boxes = []
    for i in range(len(bbss_aug)):
        bbs_aug = bbss_aug[i].bounding_boxes
        image_boxes = []
        for bbs in bbs_aug:
            x1 = bbs.x1 / image_width
            x2 = bbs.x2 / image_width
            y1 = bbs.y1 / image_height
            y2 = bbs.y2 / image_height
            box = [x1, y1, x2, y2]
            final_box = np.minimum(1, np.maximum(0, box))
            image_boxes.append(final_box)
        all_images_boxes.append(image_boxes)

    return images_aug, all_images_boxes


def save(output_dir, images, bboxes):
    label_dir = output_dir + "/labels"
    data_dir = output_dir + ("/data" if bboxes else "")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    global file_sequence
    for i in range(len(images)):
        image = images[i]
        if bboxes:
            bbox = bboxes[i]
            for bb in bbox:
                label_file = '{}/{}.txt'.format(label_dir, file_sequence)
                f = open(label_file, 'w')
                f.write("1 {}\n".format(" ".join([str(x) for x in bb])))
                f.close()
                print(label_file)
        filename = "{}/{}.jpg".format(data_dir, file_sequence)
        print(filename)
        cv2.imwrite(filename, image)
        file_sequence += 1


def augment_yolo():
    output_dir = "../detection_data_augmented"
    input_dir = "../detection_data"
    data_dir = input_dir + "/data"
    labels_dir = input_dir + "/labels"
    images = glob.glob(data_dir + "/*.jpg")
    labels = glob.glob(labels_dir + "/*.txt")
    shutil.rmtree(output_dir)
    for i in range(len(images)):
        image = images[i]
        label_file = labels[i]
        with open(label_file, 'r') as lb_file:
            bboxes = []
            for label in lb_file.readlines():
                bbox = np.asarray(label.split(" ")[1:]).astype(np.float)
                bboxes.append(bbox)
            read_image = cv2.imread(image)
            augmented_images, augmented_bboxes = augment_image(read_image, bboxes)
            save(output_dir, augmented_images, augmented_bboxes)


def augment_classifier():
    global file_sequence
    output_dir = "../recognition/augmented_data"
    input_dir = "../recognition/data"

    shutil.rmtree(output_dir)

    labels = glob.glob(input_dir + "/*")
    for label in labels:
        label_files = glob.glob(label + "/*.jpg")
        dir_label_class = output_dir + "/" + label.split("\\")[-1]
        for image in label_files:
            read_image = cv2.imread(image)
            augmented_images, _ = augment_image(read_image, [])
            save(dir_label_class, augmented_images, None)
        file_sequence = 1


if __name__ == '__main__':
    # augment_yolo()
    augment_classifier()
