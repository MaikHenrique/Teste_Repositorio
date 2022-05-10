import os
import cv2
import glob

import cv2

loaded_files = {}

class CompareImage(object):

    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 1
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path

    def compare_image(self):
        image_1 = self.get_image(self.image_1_path)
        image_2 = self.get_image(self.image_2_path)
        # image_2 = cv2.imread(self.image_2_path, 0)
        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            return commutative_image_diff
        return 10000 # random failure value

    def get_image(self, image_path):
        if image_path in loaded_files:
            return loaded_files.get(image_path)
        else:
            readed_image = cv2.imread(image_path, 0)
            loaded_files[image_path] = readed_image
            return readed_image

    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff


def diff(li1, li2):
    return list(set(li1) - set(li2))


if __name__ == '__main__':
    base_dir = "C:\\data\\"
    # files = glob.glob(base_dir + "**\\*.jpg")
    outdir = base_dir + "data_new\\"
    os.mkdir(outdir)
    start_index = 128
    # for i, file in enumerate(files):
    #     os.rename(file, outdir + str(start_index + i) + ".jpg")

    # images_dir = "C:\\data\\data\\"
    # files = glob.glob(images_dir + "*.jpg")
    # # print(CompareImage(images_dir + "131.jpg", images_dir + "151.jpg").compare_image())
    # deleted_files = []
    # for file in files:
    #     if file not in deleted_files:
    #         for file2 in files:
    #             if file != file2 and file2 not in deleted_files:
    #                 print("Comparando: ", file, file2)
    #                 if CompareImage(file, file2).compare_image() < 0.25:
    #                     deleted_files.append(file2)
    #
    # print(len(deleted_files))
    # print(diff(files, deleted_files))
    files = ['C:\\data\\data\\153.jpg', 'C:\\data\\data\\187.jpg', 'C:\\data\\data\\559.jpg', 'C:\\data\\data\\483.jpg',
     'C:\\data\\data\\308.jpg', 'C:\\data\\data\\940.jpg', 'C:\\data\\data\\294.jpg', 'C:\\data\\data\\933.jpg',
     'C:\\data\\data\\128.jpg', 'C:\\data\\data\\133.jpg', 'C:\\data\\data\\543.jpg', 'C:\\data\\data\\216.jpg',
     'C:\\data\\data\\950.jpg', 'C:\\data\\data\\130.jpg', 'C:\\data\\data\\803.jpg', 'C:\\data\\data\\248.jpg',
     'C:\\data\\data\\139.jpg', 'C:\\data\\data\\574.jpg']
    for i, file in enumerate(files):
        os.rename(file, outdir + str(start_index + i) + ".jpg")