import cv2
import imutils
import numpy as np


def greyscale():
    img = cv2.imread('../data/PSTTWM.png')
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('grey', grey_img)
    cv2.waitKey(0)
    ret, mask = cv2.threshold(grey_img, 0, 0, cv2.THRESH_BINARY)
    img2_fg = cv2.bitwise_and(grey_img, grey_img, mask=mask)
    cv2.imshow("img2_fg", img2_fg)
    cv2.waitKey(0)

    cv2.imwrite("../data/PSTTWM_grey.png", grey_img)

def remove_transparency(source, background_color):
    source_img = cv2.cvtColor(source[:,:,:3], cv2.COLOR_BGR2GRAY)
    source_mask = source[:,:,3] * (1 / 255.0)

    background_mask = 1.0 - source_mask

    bg_part = (background_color * (1 / 255.0)) * (background_mask)
    source_part = (source_img * (1 / 255.0)) * (source_mask)

    return np.uint8(cv2.addWeighted(bg_part, 255.0, source_part, 255.0, 0.0))

def add_background(filePath, background=176):
    image = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
    gray_image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    # image = remove_transparency(image, 176)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    alpha = image[:,:,3]* (1 / 255.0)
    beta = (1.0 - alpha);
    fg_img = gray_image * alpha
    bg_img = (255) * beta
    dst = np.uint8(cv2.addWeighted(bg_img, 1, fg_img, 1, 0.0))

    # trans_mask = alpha == 0
    # image[trans_mask] = [255, 255, 255, 255]
    # foreground = cv2.multiply(alpha/255.0, image[:,:,0:3])
    cv2.imshow('grey', dst)
    cv2.waitKey(0)

    cv2.imwrite('../data/SSCPMT_concat_contour_new_bg.png', dst)

#
# rotated = imutils.rotate(img, 15)
# cv2.imshow('rotate img', rotated)
# cv2.waitKey(0)


def playground():
    # greyscale()
    add_background('../data/SSCPMT_concat_contour_90x35.png')
    pass


if __name__ == '__main__':
    playground()
    cv2.destroyAllWindows()
