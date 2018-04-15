from numpy import *
from sklearn.cluster import KMeans
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


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


def load_img(img, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(img, flag)
    # return cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    return cv2.resize(img, (600, 800))


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
    coord_array = np.empty([s[0], s[1], 2], dtype=uint8)
    for i in range(s[0]):
        for j in range(s[1]):
            coord_array[i, j] = [i, j]
    return np.append(img, coord_array * scale, 2)


# def features_remove_coord(img):
#     return np.delete(img, [3, 4], 2)


def k_means_clustering(data, n_clusters, max_iter, n_init):
    clf = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    clf.fit(data)
    return clf.labels_, clf.cluster_centers_


def scale_range(input, in_min, in_max, out_min, out_max):
    return ((input - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min;


def generate_labelled_img(labels, shape):
    result = np.empty(shape, dtype=uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # scaled_value = scale_range(labels[i * shape[1] + j], 0, k-1, 0, 255)
            # rgb_color = np.uint8([[[0, scaled_value, 0]]])
            color_code = cluster_color_dict(labels[i * shape[1] + j])[1]
            rgb_color = np.uint8([[color_code]])
            result[i, j] = cvt_gbr_2_hsv(rgb_color)[0, 0]
    return result


def display_img(img):
    plt.figure()
    plt.imshow(mpimg.imread(img))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def select_foreground(k):
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
    result = np.zeros(shape, dtype=uint8)
    img_size = [shape[0], shape[1]]
    for i in range(shape[0]):
        for j in range(shape[1]):
            if is_foreground(i, j, foreground_clusters, labels, img_size):
                result[i, j] = [255, 255, 255]
    return result


def foreground_mask(masked_img_gray, img_size):
    mask = np.empty(img_size, dtype=bool)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            mask[i, j] = (masked_img_gray[i, j] == 0)
    return mask


def reduce_noise(img, h=10):
    return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize=7, searchWindowSize=21)


def generate_img_contoured_foreground(img, h):
    img = reduce_noise(cvt_gbr_2_grayscale(img), h)
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(img)
    cv2.drawContours(out, contours, -1, 255, thickness=-1)
    out = reduce_noise(out, h)
    return out


def separate_foreground(img, k=8, coord_features_scale=0.25, clustering_max_iter=2000, clustering_n_init=10, h=50):
    img_hsv = cvt_gbr_2_hsv(load_img(img))
    s = img_hsv.shape

    img_hsv = features_append_coord(img_hsv, coord_features_scale)
    data = img_hsv.reshape(s[0] * s[1], s[2] + 2)

    labels, centroids = k_means_clustering(data, k, max_iter=clustering_max_iter, n_init=clustering_n_init)

    labelled_img = generate_labelled_img(labels, s)
    labelled_img = cvt_hsv_2_gbr(labelled_img)

    cv2.imwrite('result.png', labelled_img)

    display_img('result.png')
    foreground_clusters = select_foreground(k)

    masked_img = generate_masked_img(foreground_clusters, labels, s)

    masked_img_gray = generate_img_contoured_foreground(masked_img, h)

    cv2.imwrite('masked_result.png', masked_img_gray)
    display_img('masked_result.png')

    return foreground_mask(masked_img_gray, [s[0], s[1]])


if __name__ == '__main__':
    separate_foreground('../data/2.jpeg')
