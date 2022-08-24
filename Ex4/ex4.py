"""
Author: Hao Zheng

Exercise 4
"""

import os
import glob
import numpy as np
import PIL
from PIL import Image, ImageStat
import pickle

"""
• image_array: A numpy array of shape (M, N, 3) and numeric datatype, which contains the RGB image data.
• offset: A tuple containing 2 int values. These two values specify the offset of the first grid point in x and y direction.
offset[0] -> horizontal offset
offset[1] -> vertical offset
• spacing: A tuple containing 2 int values. These two values specify the spacing between
two successive grid points in x and y direction. Please see Figure 1 below for more information.
spacing[0] -> space between horizontal grid points
spacing[1] -> space between vertical grid points


(H, W, 3)

"""


def ex4(image_array, offset, spacing):
    # TODO implement ckecks for
    # check of valid inputs
    if not (isinstance(image_array, np.ndarray)):
        raise TypeError("image_array is not a numpy array")
    if image_array.ndim != 3:
        raise NotImplementedError("image_array is not a 3D array.")
    if image_array.shape[2] != 3:
        raise NotImplementedError("image_array the size of the 3rd dimension is not equal to 3")

    try:
        offset1, offset2 = int(offset[0]), int(offset[1])
        spacing1, spacing2 = int(spacing[0]), int(spacing[1])

    except ValueError:
        raise ValueError('Offset and/or spacing values are not convertible to integers')

    if not all(x >= 0 for x in offset) or not all(x <= 32 for x in offset):
        raise ValueError("The values in offset are smaller than 0 or larger than 32.")

    if not all(x >= 2 for x in spacing) or not all(x <= 8 for x in spacing):
        raise ValueError("The values in spacing are smaller than 2 or larger than 8.")

    offset_x, offset_y = offset[0], offset[1]
    spacing_x, spacing_y = spacing[0], spacing[1]
    img_array_copy = image_array.copy()
    # print(f"img shape {image_array.shape}")


    # create input_array
    input_array = np.zeros_like(img_array_copy, dtype=img_array_copy.dtype)

    count_valid = 0

    for y in range(offset_y, img_array_copy.shape[0], spacing_y):
        for x in range(offset_x, img_array_copy.shape[1], spacing_x):
            input_array[y, x] = img_array_copy[y, x]
            count_valid += 1
    # TODO count remaining pixels if < 144 -> ValueError

    input_array = np.transpose(input_array, (2, 0, 1))


    if count_valid < 144:
        raise ValueError("The number of the remaining known image pixels would be smaller than 144")

    # create known_array
    known_array = np.zeros_like(img_array_copy, dtype=img_array_copy.dtype)

    for y in range(offset_y, img_array_copy.shape[0], spacing_y):
        for x in range(offset_x, img_array_copy.shape[1], spacing_x):
            known_array[y, x] = 1

    known_array = np.transpose(known_array, (2, 0, 1))

    # create target_array
    # target_mask = np.where((img_array_copy == 0) | (img_array_copy == 1), True, False)
    target_array = np.transpose(img_array_copy, (2, 0, 1))[known_array < 1]

    return input_array, known_array, target_array
