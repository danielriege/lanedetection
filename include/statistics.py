#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

from . import supervisely_parser as sp
from . import logger

# calculate the ratio of pixels per class
def calc_class_distribution_data(dataset, class_color_description):
    num_pixels_per_class = {} # dict with class name as key and number of pixels as value
    num_images_per_class = {} # dict with class name as key and number of images as value
    total_num_pixels = 0
    # check if dataset is a list of path names
    if not isinstance(dataset, list):
        logger.printe("Dataset is not a list of path names")
        return None
    # show progress bar to 100%
    for img in tqdm(dataset, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc="Calculating class distribution data"):
        img_, masks = sp.parse_image(img[0], img[1], class_color_description, None)
        if total_num_pixels == 0:
            total_num_pixels = img_.shape[0] * img_.shape[1] * len(dataset)
        for class_name, mask in masks.items():
            if class_name not in num_pixels_per_class:
                num_pixels_per_class[class_name] = 0
            num_pixels_in_class = np.sum(mask/255)
            num_pixels_per_class[class_name] += num_pixels_in_class
            if num_pixels_in_class > 0:
                if class_name not in num_images_per_class:
                    num_images_per_class[class_name] = 0
                num_images_per_class[class_name] += 1
    return (num_pixels_per_class, num_images_per_class, total_num_pixels)