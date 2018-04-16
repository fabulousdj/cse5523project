import numpy as np
from numpy import *
import cv2
import foreground_separation.ImgProcessingHelper as img_helper
import foreground_separation.ClusteringHelper as clustering_helper


def cluster_color_dict(k):
    # b, g, r
    dict = [
        ['red', [0, 0, 225]],
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


def scale_range(input, in_min, in_max, out_min, out_max):
    return ((input - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min;


def generate_labelled_img(labels, shape):
    out = np.empty(shape, dtype=uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            color_code = cluster_color_dict(labels[i * shape[1] + j])[1]
            rgb_color = np.uint8([[color_code]])
            out[i, j] = img_helper.cvt_gbr_2_hsv(rgb_color)[0, 0]
    return out


def foreground_selection(k):
    print('Please select the foreground. Use white space to separate the numbers.')
    for i in range(k):
        print('%d. %s' % (i, cluster_color_dict(i)[0]))
    return [int(x) for x in input().split()]


def is_point_in_foreground(row, col, foreground_clusters, labels, img_size):
    upper = row
    lower = row
    left = col
    right = col

    while upper >= 0 and labels[upper * img_size[1] + col] not in foreground_clusters:
        upper -= 1
    while lower < img_size[0] and labels[lower * img_size[1] + col] not in foreground_clusters:
        lower += 1
    while left >= 0 and labels[row * img_size[1] + left] not in foreground_clusters:
        left -= 1
    while right < img_size[1] and labels[row * img_size[1] + right] not in foreground_clusters:
        right += 1

    return upper >= 0 and lower < img_size[0] and left >= 0 and right < img_size[1]


def is_foreground(row, col, foreground_clusters, labels, img_size):
    return labels[row * img_size[1] + col] in foreground_clusters


def generate_masked_img(foreground_clusters, labels, shape):
    out = np.zeros(shape, dtype=uint8)
    img_size = [shape[0], shape[1]]
    for i in range(shape[0]):
        for j in range(shape[1]):
            if is_foreground(i, j, foreground_clusters, labels, img_size):
                out[i, j] = [255, 255, 255]
    return out


# edge_len must be odd
def get_block(row, col, matrix, edge_len):
    size = matrix.shape
    upper = row - edge_len // 2
    lower = row + edge_len // 2
    left = col - edge_len // 2
    right = col + edge_len // 2

    if upper < 0:
        upper = 0
    if lower >= size[0]:
        lower = size[0] - 1
    if left < 0:
        left = 0
    if right >= size[1]:
        right = size[1] - 1

    return matrix[upper:lower+1, left:right+1].copy()


def smoothing_masked_img(masked_img_gray, img_size):
    out = np.zeros(img_size, dtype=np.uint8)
    block_edge_len = 5
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            block = get_block(i, j, masked_img_gray, edge_len=15)
            block_size = block.shape
            out[i, j] = (sum(sum(arr) for arr in block) - masked_img_gray[i, j]) // (block_size[0] * block_size[1] - 1)
    return out


def generate_img_contoured_foreground(img, h):
    img = img_helper.reduce_noise(img_helper.cvt_gbr_2_grayscale(img), h)
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(img)
    cv2.drawContours(out, contours, -1, 255, thickness=-1)
    out = img_helper.reduce_noise(out, h)
    return out


def separate_foreground(img, k=8, coord_features_scale=0.25, clustering_max_iter=1000, clustering_n_init=10, denoise_level=50):
    img_hsv = img_helper.cvt_gbr_2_hsv(img)
    s = img_hsv.shape

    img_hsv = img_helper.features_append_coord(img_hsv, coord_features_scale)

    labels = clustering_helper.k_means_clustering(
        img_hsv,
        shape=[s[0], s[1], s[2] + 2],
        n_clusters=k,
        max_iter=clustering_max_iter,
        n_init=clustering_n_init)

    labelled_img = generate_labelled_img(labels, s)
    labelled_img = img_helper.cvt_hsv_2_gbr(labelled_img)

    img_helper.save_img(labelled_img, 'result.png')
    img_helper.display_img_file('result.png')

    foreground_clusters = foreground_selection(k)
    masked_img = generate_masked_img(foreground_clusters, labels, s)
    masked_img_gray = generate_img_contoured_foreground(masked_img, denoise_level)
    masked_img_gray = smoothing_masked_img(masked_img_gray, [s[0], s[1]])

    img_helper.save_img(masked_img_gray, 'masked_result.png')
    img_helper.display_img_file('masked_result.png')

    return masked_img_gray


if __name__ == '__main__':
    img = img_helper.load_img('../data/selfie.jpeg', [600, 800])
    # separate_foreground('../data/selfie.jpeg')
    separate_foreground(img, k=8)
