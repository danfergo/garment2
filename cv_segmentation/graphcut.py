import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

color_mask = np.full((341, 426, 3), 0, np.uint8)


def depth_list(path, depth):
    dir_list = os.listdir(path)
    if depth == 0:
        return map(lambda f: path + '/' + f, dir_list)
    else:
        deep_list = []
        for p in dir_list:
            deep_list += depth_list(path + '/' + p, depth - 1)
        return deep_list


def on_click(ev, x, y, flags, user_data):
    isLeftClicking = flags & 1
    isRightClicking = (flags & 2) >> 1

    if isLeftClicking:
        cv2.circle(color_mask, (x, y), 0, (0, 0, 255), 3)
    elif isRightClicking:
        # cv2.GC_FGD
        cv2.circle(color_mask, (x, y), 0, (0, 255, 0), 3)

    if ev == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (" + str(x) + ", " + str(y) + ")")
    elif ev == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (" + str(x) + ", " + str(y) + ")")
    elif ev == cv2.EVENT_RBUTTONUP:
        print("Right button of the mouse is clicked - position (" + str(x) + ", " + str(y) + ")")
    elif ev == cv2.EVENT_MBUTTONDOWN:
        print("Middle button of the mouse is clicked - position (" + str(x) + ", " + str(y) + ")")
    elif ev == cv2.EVENT_MOUSEMOVE:
        print("Mouse move over the window - position (" + str(x) + ", " + str(y) + ")")
        # print(str(ev) + ' ' + str(x) + ' ' + str(y))


#
# def nothing(x):
#     pass
def main():
    global color_mask
    cv2.namedWindow("img", 1)
    cv2.setMouseCallback("img", on_click)

    files = depth_list('../data/clopema', 2)
    print(files)
    for f in files:

        while 1:
            img = cv2.imread(f)
            #
            # while 1:
            #     cv2.imshow('img', img)
            #     cv2.imshow('mask', mask)
            #     k = cv2.waitKey(10) & 0xFF
            #     if k == 27:
            #         break
            #
            w, h = img.shape[:2]

            # mask = np.zeros((h, w, 3), np.uint8)

            # color_mask[1, 1] = cv2.GC_PR_BGD
            # color_mask[0, h - 1] = cv2.GC_PR_BGD
            # color_mask[w - 1, 0] = cv2.GC_PR_BGD
            # color_mask[w - 1, h - 1] = cv2.GC_PR_BGD

            color_mask[100, 100] = 150
            color_mask[100, h - 100] = 150
            color_mask[w - 100, 100] = 150
            color_mask[w - 100, h - 100] = 150
            color_mask[w / 2, h / 2] = 255

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            mask = np.full(color_mask.shape, cv2.GC_PR_BGD, np.uint8)
            mask[color_mask == (0, 0, 0)] = cv2.GC_PR_BGD
            mask[color_mask == 255] = cv2.GC_FGD
            mask[color_mask == 150] = cv2.GC_PR_FGD
            mask[color_mask == 100] = cv2.GC_BGD

            cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            img = img * mask2[:, :, np.newaxis]
            cv2.rectangle(img, (0, 0), (h - 1, w - 1), (0, 0, 255))

            cv2.imshow('img', img + color_mask)
            cv2.imshow('mask', mask)
            cv2.waitKey(5)
            # plt.imshow(img), plt.colorbar(), plt.show()


if __name__ == '__main__':
    main()
