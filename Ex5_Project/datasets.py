"""
Author: Hao Zheng
Matr.Nr.: K01608113
Exercise 5
"""

import glob
import os
import random

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

'''
Classes used for our datasets/loader and the provided function from ex4 to create input array, known arrray and target arrays for training data
The number of images used for training/validation/test data can be adjusted here, since I do not have the resources to use the whole image set ( ~ 29k images),
I used only 5 000 images for training my model. 
'''


# use ex4 function for reading training data and creating inputs and target images/arrays
MIN_OFFSET = 0
MAX_OFFSET = 8
MIN_SPACING = 2
MAX_SPACING = 6
MIN_KNOWN_PIXELS = 144

im_shape = 100
resize_transforms = transforms.Compose([
    transforms.Resize(size=im_shape),
    transforms.CenterCrop(size=(im_shape, im_shape)),
])

random.seed(10)



def create_inputs_targets(image_array: np.ndarray, offset: tuple, spacing: tuple):
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be a numpy array!")

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise NotImplementedError("image_array must be a 3D numpy array whose 3rd dimension is of size 3")

    # Check for conversion to int (would raise ValueError anyway, but we will write a nice error message)
    try:
        offset = [int(o) for o in offset]
        spacing = [int(s) for s in spacing]
    except ValueError as e:
        raise ValueError(f"Could not convert entries in offset and spacing ({offset} and {spacing}) to int! Error: {e}")

    for i, o in enumerate(offset):
        if o < MIN_OFFSET or o > MAX_OFFSET:
            raise ValueError(f"Value in offset[{i}] must be in [{MIN_OFFSET}, {MAX_OFFSET}] but is {o}")

    for i, s in enumerate(spacing):
        if s < MIN_SPACING or s > MAX_SPACING:
            raise ValueError(f"Value in spacing[{i}] must be in [{MIN_SPACING}, {MAX_SPACING}] but is {s}")

    # Change dimensions from (H, W, C) to PyTorch's (C, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))

    # Create known_array
    known_array = np.zeros_like(image_array)
    known_array[:, offset[1]::spacing[1], offset[0]::spacing[0]] = 1

    known_pixels = np.sum(known_array[0], dtype=np.uint32)
    if known_pixels < MIN_KNOWN_PIXELS:
        raise ValueError(f"The number of known pixels after removing must be at "
                         f"least {MIN_KNOWN_PIXELS} but is {known_pixels}")

    # Create target_array - don't forget to use .copy(), otherwise target_array
    # and image_array might point to the same array!
    target_array = image_array[known_array == 0].copy()
    # whole_target_image_array = image_array.copy()
    # Use image_array as input_array
    image_array[known_array == 0] = 0

    return image_array, known_array, target_array


class new_ImageDataset():
    def __init__(self, image_dir):
        # get filepaths from images
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))

        # n images can be adjusted here, if all images should be used, comment out the following line
        self.image_files = self.image_files[:500]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_data = self.image_files[idx]
        return image_data, idx


class new_GridImageSet():
    def __init__(self, dataset):
        self.image_files_paths = dataset

    def __getitem__(self, index):
        image_path = self.image_files_paths[index][0]
        image = Image.open(image_path)
        # resize images to 100 x 100
        resize_transforms = transforms.Compose([
            transforms.Resize(size=(im_shape, im_shape))
        ])
        image = resize_transforms(image)
        # image = Image.open(image_path)
        image_as_array = np.array(image, dtype=np.float32)
        image_as_array = image_as_array / 255

        target_image = image_as_array.copy()
        # randomize spacings & offsets

        spacings = (random.randint(MIN_SPACING, MAX_SPACING), random.randint(MIN_SPACING, MAX_SPACING))
        offsets = (random.randint(MIN_OFFSET, MAX_OFFSET), random.randint(MIN_OFFSET, MAX_OFFSET))
        input_array, known_array, target_array = create_inputs_targets(image_as_array, offset=offsets, spacing=spacings)

        # convert numpy arrays  ot tensors
        input_array = torch.from_numpy(input_array)
        # reshape known_array to (1,H,W)
        known_array = torch.from_numpy(np.array([known_array[0]]))
        # transpose whole image array to tensor shape 3,H,W
        targets = np.transpose(target_image, (2, 0, 1))
        targets = torch.from_numpy(targets)

        # target_array = torch.from_numpy(target_array)
        # concate input arry with 3 channel with known array( 1 channel9 -> (4, 100, 100)
        inputs_concatenated = torch.concat((input_array, known_array), dim=0)

        # returns input arrays, target array and index
        return inputs_concatenated, targets, index

    def __len__(self):
        return len(self.image_files_paths)


'''
Initially, I planned to use a custom collate function but later realized that it is not necessary if all my input images are resized to the same dimension 100x100.
Therefore, the code was commented since it is not a complete collate function.
'''

# def new_collate_func(batch_list: list):
#     # create zero tensor of shape (n_samples, n_features, max_X, max_Y) for stacked input arrays
#     #
#     # copy input values in zero tensor
#
#     # get image_arrays in list
#     batch_images = [sample[0] for sample in batch_list]
#
#     n_samples = len(batch_list)
#     # n_features = batch_images[0].shape[0]
#     n_features = 3
#     max_X = np.max([batch_img.shape[1] for batch_img in batch_images])
#     # max_X = 100
#     max_Y = np.max([batch_img.shape[2] for batch_img in batch_images])
#     # max_Y = 100
#     stacked_inputs = torch.zeros(size=(n_samples, n_features, max_X, max_Y), dtype=torch.float32)
#
#     # copy input values in stacked_inputs
#     # for i, (inputs_concatenated) in enumerate(batch_images):
#     for i, (input_array, target_array, index) in enumerate(batch_list):
#         # inputs_concatenated = torch.from_numpy(inputs_concatenated)
#         stacked_inputs[i] = input_array
#
#     return stacked_inputs
