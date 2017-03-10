# /home/danfergo/SIG/Datasets/CloPeMa/CTU/ColorAndDepth/FlatAndWrinkled
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
import random
from PIL import Image

import os
import shutil
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import yaml

WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX
FW = 1
LH = 25
FS = 0.6

PATH_FLAT_N_WRINKLED = '../../../Datasets/CloPeMa/CTU/ColorAndDepth/FlatAndWrinkled/'
PATH_FOLDED = '../../../Datasets/CloPeMa/CTU/ColorAndDepth/Folded/'

OUT_PATH = '../data/clopema/'
OUT_TRAIN_PATH = OUT_PATH + 'train/'
OUT_VALIDATION_PATH = OUT_PATH + 'validation/'

VALIDATION_PERCENT = 0.1
RS_F = 3


valid_counter = 0
counter_per_cat = {}

threshold = 1160

def main():
    clean_folder()
    read_raw_dataset()
    show_stats()


def clean_folder():
    random.seed()

    shutil.rmtree(OUT_PATH)
    os.makedirs(OUT_TRAIN_PATH)
    os.makedirs(OUT_VALIDATION_PATH)


def setTreshold(x):
    pass
    # threshold = x


# cv2.namedWindow('image')
# cv2.createTrackbar('threshold', 'image', 0, 200, setTreshold)


def read_raw_dataset():
    global valid_counter, threshold

    for x in range(1, 3332):
            threshold = cv2.getTrackbarPos('threshold', 'image') + 1150
            # i = 3
            path = PATH_FLAT_N_WRINKLED if x < 2051 else PATH_FOLDED
            #
            # if x > 10:
            #     break
            try:
                file_name = 'cloA' + str(x).zfill(5) + '.yaml'
                meta_file = open(path + file_name)
                meta_data = yaml.load(meta_file)

                img_file_name = meta_data['path_c']
                # depth_file_name = meta_data['path_d']
                img_category = meta_data['type']
                img_poly = meta_data['poly_c']

                meta_file = open(path + file_name)
                meta_data = yaml.load(meta_file)

                # depth_img = cv2.imread(path + depth_file_name, -1)
                # if depth_img is None:
                #     continue
                #
                # x00 = depth_img[0+30, 0+20]
                # x11 = depth_img[479-30, 639-40]
                # x01 = depth_img[0+30, 639-40]
                # x10 = depth_img[479-30, 0+20]
                #
                # v00 = np.array([0+30, 0+20, x00-1000], dtype=np.float64)
                # v11 = np.array([479-30, 639-40, x11-1000], dtype=np.float64)
                # v01 = np.array([0+30, 639-40, x01-1000], dtype=np.float64)
                # v10 = np.array([479-30, 0+20, x10-1000], dtype=np.float64)

                # print str(0+30) + ' ' + str(0+20) + ' ' + str(x00)
                # print str(479-30) + ' ' + str(0+20) + ' ' + str(x11)
                # print str(0+30) + ' ' + str(0+20) + ' ' + str(x01)
                # print str(479-30) + ' ' + str(0+20) + ' ' + str(x10)

                # v = v10 - v00
                # h = v11 - v10
                # p = np.cross(v, h)
                #
                # d = -1 * np.dot(p, v00)
                #
                # pl = np.array([p[0], p[1], p[2], d], dtype=np.float64)

                # print pl
                # print "--------------------"
                # depth_img_pil = load_img(path + depth_file_name)
                # depth_img = img_to_array(depth_img_pil)
                # h, w = depth_img.shape
                # 640, 480
                # print depth_img.shape
                # segmentation_mask = np.zeros((h, w, 3), np.uint8)

                # print threshold
                # for i in range(0, h):
                #     for j in range(0, w):
                #         # print int(depth_img[i, j])
                #         pp = np.array([i, j, depth_img[i, j]-1000, 1], dtype=np.float64)
                #         r = np.dot(pl, pp)
                #         print r
                #
                #         if 0 > r:
                #             segmentation_mask[i, j] = (255, 255, 255)
                #         else:
                #             segmentation_mask[i, j] = (0, 0, 0)
                #         # print "--------------"
                #         # if not (depth_img[xx, yy][0] == 255
                #         #         and depth_img[xx, yy][1] == 255
                #         #         and depth_img[xx, yy][2] == 255)\
                #         #     and not (depth_img[xx, yy][0] == 0
                #         #              and depth_img[xx, yy][1] == 0
                #         #              and depth_img[xx, yy][2] == 0):
                #         #     print depth_img[xx, yy]

                # print depth_img[479, 639]

                img_pil = load_img(path + img_file_name)
                img_pil = img_pil.resize((1280/RS_F, 1024/RS_F), Image.ANTIALIAS)

                img = img_to_array(img_pil)

                if img_category in counter_per_cat:
                    counter_per_cat[img_category] += 1
                else:
                    counter_per_cat[img_category] = 1
                valid_counter += 1

                path = OUT_VALIDATION_PATH if random.random() < VALIDATION_PERCENT else OUT_TRAIN_PATH
                if not os.path.exists(path + img_category):
                    os.makedirs(path + img_category)

                img_pil.save(path +
                         '/' + img_category +
                         '/' + str(counter_per_cat[img_category]).zfill(4) + '.jpeg')

                xx = img.astype(np.uint8)
                for m in img_poly:
                    cv2.circle(xx, (int(m[0]/RS_F), int(m[1]/RS_F)), 5, (0, 0, 255), 1)
                # cv2.circle(segmentation_mask, (int(0+20), int(0+30)), 5, (0, 0, 255), 1)
                # cv2.circle(segmentation_mask, (int(0+20), int(479-30)), 5, (0, 0, 255), 1)
                # cv2.circle(segmentation_mask, (int(639-40), int(479-30)), 5, (0, 0, 255), 1)
                # cv2.circle(segmentation_mask, (int(639-40), int(0+30)), 5, (0, 0, 255), 1)

                cv2.imshow("image", xx)
                cv2.setWindowTitle("image",
                                   'Ttl: ' + str(valid_counter).zfill(4).ljust(10) +
                                   'Cat: ' + img_category.ljust(15) +
                                   '/cat: ' + str(counter_per_cat[img_category]).zfill(3).ljust(6))
                cv2.waitKey(1)

            except Exception as e:
                print str(e)
                continue



def show_stats():
    stats_frame = np.zeros((1024/RS_F, 1280/RS_F), dtype=np.uint8)

    i = 1
    for key in counter_per_cat:
        cv2.putText(stats_frame, key, (10, i*LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
        cv2.putText(stats_frame, str(counter_per_cat[key]), (100, i*LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
        i += 1
    cv2.putText(stats_frame, 'TOTAL', (10, (i+1) * LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
    cv2.putText(stats_frame, str(valid_counter), (100, (i+1) * LH), FONT, FS, WHITE, FW, cv2.LINE_AA)

    cv2.imshow("image", stats_frame)
    cv2.setWindowTitle("image", "Stats")
    cv2.waitKey(10000)


if __name__ == '__main__':
    main()



# x = x.reshape((1,) + x.shape)
# i = 0
# for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#         i += 1
# cv2.imshow("image", batch[0].astype(np.uint8))
# cv2.waitKey(2)

# if i > 2:
# break  # otherwise the generator would loop indefinitely
