import scipy.io
import numpy as np
from numpy import *
from sklearn.cluster import KMeans
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def cluster_color_dict(k):
    # b, g, r
    dict = [
        ['red', [0, 0, 204]],
        ['green', [0, 255, 0]],
        ['blue', [255, 0, 0]],
        ['white', [255, 255, 255]],
        ['yellow', [0, 255, 255]],
        ['cerise', [255, 0, 255]],
        ['black', [0, 0, 0]],
        ['orange', [51, 153, 255]],
        ['purple', [102, 0, 102]],
        ['cyan', [204, 204, 51]]
    ]
    return dict[k]


def load_img(img, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(img, flag)
    #return cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    return cv2.resize(img, (75, 100))


def cvt_gbr_2_hsv(color):
    return cv2.cvtColor(color, cv2.COLOR_BGR2HSV)


def cvt_hsv_2_gbr(color):
    return cv2.cvtColor(color, cv2.COLOR_HSV2BGR)


def features_append_coord(img, scale):
    s = img.shape
    coord_array = np.empty([s[0], s[1], 2], dtype=uint8)
    for i in range(s[0]):
        for j in range(s[1]):
            coord_array[i, j] = [i, j]
    return np.append(img, coord_array * scale, 2)


# def features_remove_coord(img):
#     return np.delete(img, [3, 4], 2)


def k_means_clustering(data, n_clusters, n_init=10, max_iter=300):
    clf = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    clf.fit(data)
    return clf.labels_, clf.cluster_centers_


def scale_range(input, in_min, in_max,  out_min, out_max):
    return ((input - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min;


def generate_labelled_img(labels, shape):
    result = np.empty(shape, dtype=uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # scaled_value = scale_range(labels[i * shape[1] + j], 0, k-1, 0, 255)
            # rgb_color = np.uint8([[[0, scaled_value, 0]]])
            color_code = cluster_color_dict(labels[i * shape[1] + j])[1]
            rgb_color = np.uint8([[color_code]])
            result[i, j] = cvt_gbr_2_hsv(rgb_color)[0,0]
    return result


def display_img(img):
    plt.figure()
    plt.imshow(mpimg.imread(img))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def select_foreground():
    print('Please select the foreground. Use white space to separate the numbers.')
    for i in range(k):
        print('%d. %s' % (i, cluster_color_dict(i)[0]))
    return [int(x) for x in input().split()]


def is_point_in_foreground(row, col, foreground_clusters, lables, img_size):
    up = [lables[i * img_size[1] + col] for i in range(0, row)]
    down = [lables[i * img_size[1] + col] for i in range(row, img_size[0])]
    left = [lables[row * img_size[1] + j] for j in range(0, col)]
    right = [lables[row * img_size[1] + j] for j in range(col, img_size[1])]

    directions = [up, down, left, right]
    has_bounds = [False] * 4
    for i in range(len(directions)):
        has_bounds[i] = [(k in directions[i]) for k in foreground_clusters]
    return all(has_bounds)


def foreground_mask(foreground_clusters, labels, img_size):
    mask = np.empty(img_size, dtype=uint8)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            mask[i, j] = is_point_in_foreground(i, j, foreground_clusters, labels, img_size)
    return mask


def generate_masked_img(mask, shape):
    result = np.empty(shape, dtype=uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            color_code = [0, 0, 0] if (mask[i, j] == 0) else [255, 255, 255]
            result[i, j] = color_code
    return result


if __name__ == '__main__':

    img_hsv = cvt_gbr_2_hsv(load_img('../data/2.jpeg'))
    # img_hsv = enhance_contrast(img_hsv)
    s = img_hsv.shape
    img_hsv = features_append_coord(img_hsv, 0.2)
    data = img_hsv.reshape(s[0] * s[1], s[2]+2)

    # set number of clusters
    k = 8

    labels, centroids = k_means_clustering(data, k, max_iter=1000)

    result_img = generate_labelled_img(labels, s)
    result_img = cvt_hsv_2_gbr(result_img)

    # cv2.imshow('img', result_img)
    # print(cv2.waitKey())
    cv2.imwrite('result.png', result_img)

    display_img('result.png')
    foreground_clusters = select_foreground()

    mask = foreground_mask(foreground_clusters, labels, [s[0], s[1]])
    masked_img = generate_masked_img(mask, s)

    cv2.imwrite('masked_result.png', masked_img)
    display_img('masked_result.png')
    #print(foreground_clusters)

