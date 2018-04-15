import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_img(file, size, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(file, flag)
    # return cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    return cv2.resize(img, (size[0], size[1]))


def save_img(img, name):
    cv2.imwrite(name, img)


def cvt_gbr_2_hsv(color):
    return cv2.cvtColor(color, cv2.COLOR_BGR2HSV)


def cvt_hsv_2_gbr(color):
    return cv2.cvtColor(color, cv2.COLOR_HSV2BGR)


def cvt_gbr_2_grayscale(color):
    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)


def cvt_grayscale_2_gbr(color):
    return cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)


def features_append_coord(img, scale):
    s = img.shape
    coord_array = np.empty([s[0], s[1], 2], dtype=np.uint8)
    for i in range(s[0]):
        for j in range(s[1]):
            coord_array[i, j] = [i, j]
    return np.append(img, coord_array * scale, 2)


def reduce_noise(img, h=10):
    return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize=7, searchWindowSize=21)


def display_img_file(file):
    plt.figure()
    plt.imshow(mpimg.imread(file))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
