import cv2
import foreground_separation.ForegroundSeparator as fg_separator
import foreground_separation.ImgProcessingHelper as img_helper


# Image blur using GaussianBlur
def blur(img, intensity):
    return cv2.GaussianBlur(img, (intensity, intensity), 0)


def merge(blurred_img, origin_img, mask, img_size):
    out = blurred_img.copy()
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            p = mask[i, j] / 255.0
            out[i, j] = origin_img[i, j] * p + blurred_img[i, j] * (1 - p)
    return out


def background_blur(img, intensity=5):
    mask = fg_separator.separate_foreground(img, clustering_n_init=5)
    blurred_img = blur(img, intensity)
    s = img.shape
    bg_blurred_img = merge(blurred_img, img, mask, [s[0], s[1]])

    img_helper.save_img(bg_blurred_img, 'blurred_result.png')
    img_helper.display_img_file('blurred_result.png')

    return bg_blurred_img


if __name__ == '__main__':
    img = img_helper.load_img('../data/selfie.jpeg', [600, 800])
    # img = img_helper.load_img('../data/bottle.png', [600, 800])
    background_blur(img, intensity=25)
