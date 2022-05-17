import cv2
import numpy as np
import os

red_path = '../data/red/'
green_path = '../data/green/'
blue_path = '../data/blue/'
original_path = './orig/'
glared_path = '../data/imgs/'
new_path = '../data/masks/'


for pos, file in enumerate(os.listdir(red_path)):
    print(f"{pos}/{len(os.listdir(red_path))}")
    red = cv2.imread(red_path + file, cv2.IMREAD_UNCHANGED)
    green = cv2.imread(green_path + file, cv2.IMREAD_UNCHANGED)
    blue = cv2.imread(blue_path + file, cv2.IMREAD_UNCHANGED)
    merged = np.stack((blue, green, red), axis=-1)
    cv2.imwrite(new_path + file, merged)