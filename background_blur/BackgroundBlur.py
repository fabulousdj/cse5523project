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
            if mask[i, j]:
                out[i, j] = origin_img[i, j]
    return out


def background_blur(img, intensity=5):
    mask = fg_separator.separate_foreground(img)
    blurred_img = blur(img, intensity)
    s = img.shape
    return merge(blurred_img, img, mask, [s[0], s[1]])


if __name__ == '__main__':
    img = img_helper.load_img('../data/selfie.jpeg', [600, 800])
    # img = img_helper.load_img('../data/bottle.png', [600, 800])
    bg_blurred_img = background_blur(img, intensity=25)
    cv2.imshow("blurred", bg_blurred_img)
    cv2.waitKey()
