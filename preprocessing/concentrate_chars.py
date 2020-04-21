import cv2
import imutils
import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_csv('./az_1_100.csv')
samples = df.shape[0]

destFolder = '../data/concat/'

large_shape = [35, 90, 4]


def getBordered(image, width):
    bg = np.zeros(image.shape)
    _, contours, _ = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = 0
    bigcontour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggest:
            biggest = area
            bigcontour = contour
    return cv2.drawContours(bg, [bigcontour], 0, (255, 255, 255), width).astype(bool)


def generate_index(size=6):
    a = np.arange(6)
    np.random.shuffle(a)
    return np.sort(a[: np.random.randint(2, 7)])


def shakeImage(background, image, center):
    return background


def add_background(image, bg_color=200):
    gray_image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    alpha = image[:, :, 3] * (1 / 255.0)
    beta = (1.0 - alpha)
    fg_img = gray_image * alpha
    bg_img = bg_color * beta
    return np.uint8(cv2.addWeighted(bg_img, 1, fg_img, 1, 0.0))


def make_image(size=6):
    positions = generate_index()
    sample = np.random.randint(samples, size=positions.size)
    word = ''
    contour_img = np.zeros(large_shape, dtype='uint8')
    threshold_img = np.zeros(large_shape, dtype='uint8')
    for k in range(positions.size):
        row = sample[k]
        pos = positions[k]
        word += chr(ord('A') + df.iloc[row, 0].astype('uint8'))
        image_array = df.iloc[row, 1:].to_numpy(dtype='uint8').reshape(28, 28)
        # tmp = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGRA)

        _, alpha_white = cv2.threshold(image_array, 254, 255, cv2.THRESH_BINARY_INV)
        _, alpha_black = cv2.threshold(image_array, 10, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(alpha_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros([28, 28, 4], dtype='uint8')
        cv2.drawContours(canvas, contours, -1, (0, 0, 0, 255), 1)
        shake_rotate_angel = np.random.randint(-45, 45)
        shake_trans_x = np.random.randint(-8, 8)
        canvas = imutils.rotate(canvas, shake_rotate_angel)
        alpha = alpha_black & alpha_white
        # b, g, r = cv2.split(image_array)
        rgba = [image_array, image_array, image_array, alpha]
        dst = cv2.merge(rgba, 4)
        # cv2.imshow('threshold', dst)
        # cv2.waitKey(0)

        resized_contour = cv2.resize(canvas, (20, 20), interpolation=cv2.INTER_AREA)
        resized_threshold = cv2.resize(dst, (20, 20), interpolation=cv2.INTER_AREA)

        x_offset = 8 + shake_trans_x
        y_offset = pos * 14
        # l_img[0:28, 0:28] =image_array
        # print('resized_contour shape:', resized_contour.shape)
        # print('resized_threshold shape:', resized_contour.shape)

        contour_img[x_offset:(x_offset + 20), y_offset:(y_offset + 20)] = cv2.add(resized_contour,
               contour_img[x_offset:(x_offset + 20), y_offset:(y_offset + 20)])

        threshold_img[x_offset:(x_offset + 20), y_offset:(y_offset + 20)] = cv2.add(resized_threshold,
                threshold_img[x_offset:(x_offset + 20), y_offset:(y_offset + 20)])
        # cv2.imshow('', l_img)
        # cv2.waitKey(0)
    print(word)

    # cv2.imshow('threshold result', threshold_img)
    # cv2.imshow('contour result', contour_img)
    # cv2.waitKey(0)
    threshold_img = add_background(threshold_img, bg_color=np.random.randint(176, 255))
    contour_img = add_background(contour_img, bg_color=np.random.randint(176, 255))
    cv2.imwrite(destFolder + word + '_concat_contour.png', contour_img)
    cv2.imwrite(destFolder + word + '_concat_threshold.png', threshold_img)


def main():
    for i in range(10):
        make_image()


if __name__ == '__main__':
    main()
