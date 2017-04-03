from __future__ import division
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import exposure

import scipy
split_version = scipy.__version__.split('.')
if not(split_version[-1].isdigit()): # Remove dev string if present
        split_version.pop()
scipy_version = list(map(int, split_version))
new_scipy = scipy_version[0] > 0 or \
            (scipy_version[0] == 0 and scipy_version[1] >= 14)

if not new_scipy:
    print('You are using an old version of scipy. '
          'Active contours is implemented for scipy versions '
          '0.14.0 and above.')


def depth_list(path, depth):
    dir_list = os.listdir(path)
    if depth == 0:
        return map(lambda f: path + '/' + f, dir_list)
    else:
        deep_list = []
        for p in dir_list:
            deep_list += depth_list(path + '/' + p, depth-1)
        return deep_list


# def calc_bg():
#     files = depth_list('../data/clopema', 2)
#     counter = 1
#     bg = np.zeros((341, 426, 3), dtype=np.uint8)
#
#     for f in files:
#         img = cv2.imread(f)
#         # print img.shape, img.dtype
#
#         percent = (1 / counter)
#         bg = cv2.addWeighted(bg, 1 - percent, img, percent, 0)
#
#         cv2.imshow('image', img)
#         cv2.imshow('background', bg)
#         cv2.waitKey(1)
#         counter += 1
#
#     cv2.imwrite('bg.jpeg', bg)
#     cv2.waitKey(0)


def points(si, sj, step_i, step_j, n_elems):
    pts = []
    for x in range(n_elems):
        pts.append([si + x * step_i, sj + x * step_j])
    return pts

def nothing(x):
    pass


def main():
    # calc_bg()

    # bg = cv2.imread('bg.jpeg')
    # bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    files = depth_list('../data/clopema', 2)
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))

    w = 426
    h = 341

    # init = np.array(
    #     points(w - 2, h - 2, 0, -(h-4)/200, 200)
    #     + points(w - 2, 0 + 2, -(w-4)/200, 0, 200)
    #     + points(0 + 2, 0 + 2, 0, (h-4)/200, 200)
    #     + points(0 + 2, h - 2, (w-4)/200, 0, 200)
    #                 )

    # print init
    # print np.array([w1]).T + np.array([0, 0])


    cv2.namedWindow('image')

    cv2.createTrackbar('h1', 'image', 0, 255, nothing)
    cv2.createTrackbar('s1', 'image', 0, 255, nothing)
    cv2.createTrackbar('v1', 'image', 0, 255, nothing)

    cv2.createTrackbar('h2', 'image', 0, 255, nothing)
    cv2.createTrackbar('s2', 'image', 0, 255, nothing)
    cv2.createTrackbar('v2', 'image', 0, 255, nothing)

    for f in files:
        while 1:
            img = cv2.imread(f)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            h1 = cv2.getTrackbarPos('h1', 'image')
            s1 = cv2.getTrackbarPos('s1', 'image')
            v1 = cv2.getTrackbarPos('v1', 'image')

            h2 = cv2.getTrackbarPos('h2', 'image')
            s2 = cv2.getTrackbarPos('s2', 'image')
            v2 = cv2.getTrackbarPos('v2', 'image')

            mask = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))
            res = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow('image', img)
            cv2.imshow('mask', mask)
            cv2.imshow('res', res)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        # return

    # x = 213 + 213 * np.cos(s)
    # y = 170 + 170 * np.sin(s)
    # init = np.array([x, y]).T

    # cv2.namedWindow('image')

    # create trackbars for color change
    # cv2.createTrackbar('a', 'image', 15, 50, nothing)
    # cv2.createTrackbar('b', 'image', 47, 100, nothing)
    # cv2.createTrackbar('g', 'image', 1, 10, nothing)

    for f in files:
        img = io.imread(f, True)
        # img = clahe.apply(img)
        # img = exposure.equalize_adapthist(img)

        while 1:
            cv2.imshow('image', img)

            a = cv2.getTrackbarPos('a', 'image')
            b = cv2.getTrackbarPos('b', 'image')
            g = cv2.getTrackbarPos('g', 'image')

            snake = active_contour(gaussian(img, 3),
                                   init, alpha=a/1000, beta=b/100, gamma=g/1000)

            # fig = plt.figure(figsize=(7, 7))
            # ax = fig.add_subplot(111)
            # plt.gray()
            # ax.imshow(img)
            # ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
            # ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
            # ax.set_xticks([]), ax.set_yticks([])
            # ax.axis([0, img.shape[1], img.shape[0], 0])

            # print(snake.shape)
            img2 = np.array(img, copy=True)
            # for xy in init:
            #     # print xuy[0]
            #     cv2.circle(img2, (xy[0].astype(int), xy[1].astype(int)), 5, (0, 0, 0))

            for xy in snake:
                cv2.circle(img2, (xy[0].astype(int), xy[1].astype(int)), 5, (0, 0, 0))

            cv2.imshow('image', img2)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break

                # im = cv2.imread(f)
        # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        # # im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        #
        # # diff = im_gray.astype(np.int16) - bg_gray.astype(np.int16)
        # # diff = cv2.absdiff(im, bg)
        #
        # cv2.imshow('sub', clahe.apply(im_gray))
        # cv2.waitKey(100)



if __name__ == '__main__':
    main()