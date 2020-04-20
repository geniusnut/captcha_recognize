

import cv2
import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_csv('../data/az_1_100.csv')
samples = df.shape[0]


def make_image(size=6):
    sample = np.random.randint(samples, size=size)
    word = ''
    imgs = []
    for i in sample:
        word += chr(ord('A') + df.iloc[i, 0].astype('uint8'))
        image_array = df.iloc[i, 1:].to_numpy().reshape(28, 28)
        imgs.append(image_array)
    print(word)
    im_h = cv2.hconcat(imgs)
    cv2.imwrite(word + '_concat.png', im_h)


def main():
    for i in range(10):
        make_image()
if __name__ == '__main__':
    main()