"""
Author: Hao Zheng

Exercise 3
"""

import os
import glob
import numpy as np
import PIL
from PIL import Image, ImageStat
import pickle


# write class ImageStandardizer (in the file ex3.py). This class should provide three methods: __init__,
# analyze_images and get_standardized_images.

class ImageStandardizer:
    def __init__(self, input_dir: str):
        # scan input dir for .jpg
        if not os.path.isdir(input_dir):
            raise ValueError("Input dir ist not a valid directory")
        # self.input_dir = input_dir

        files = glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True)

        if not files:
            raise ValueError("No .jpg files in given input directory")

        files = [os.path.abspath(file) for file in files]
        files = sorted(files, reverse=False)

        self.files = files
        self.mean = np.empty(shape=3)
        self.std = np.empty(shape=3)

    def analyze_images(self):
        # calculate means & stds for each color channel - three entries for red, green, blue
        rgb_means = np.zeros(3, dtype=np.float64)
        rgb_stds = np.zeros(3, dtype=np.float64)

        for file in self.files:
            with Image.open(file) as img:
                # img_array = np.array(img)
                rgb_means = rgb_means + np.array(ImageStat.Stat(img).mean)
                rgb_stds = rgb_stds + np.array(ImageStat.Stat(img).stddev)

        rgb_means = rgb_means / len(self.files)
        rgb_stds = rgb_stds / len(self.files)

        # print(f"rgb mean: {rgb_means.dtype}, shape: {rgb_means.shape}")
        # print(f"rgb std: {rgb_stds.dtype}, shape: {rgb_stds.dtype}")
        # print(len(self.files))
        self.mean = rgb_means
        self.std = rgb_stds
        # self.mean, self.std must have array shape (3,)

        return self.mean, self.std

    def get_standardized_images(self):
        if self.mean is None or self.std is None:
            raise ValueError("Mean or Std. is None")

        for file in self.files:
            with Image.open(file, "r") as image_file:
                np_image = np.array(image_file)
                np_standardized = (np_image - self.mean) / self.std
                np_standardized = np.array(np_standardized, dtype=np.float32)
                # print(f"{file}: {np_standardized.dtype}")
                yield np_standardized
