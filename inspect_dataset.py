#!/usr/bin/env python3

import include.supervisely_parser as sp
import include.plotter as plotter
from include.logger import printe, printw, printi
import include.tensorflow.data_generator as dg

import matplotlib.pyplot as plt
import argparse
import os
import cv2

SHOW = False
PLOT = False
DG = False
RESIZE_WIDTH = None
RESIZE_HEIGHT = None
USE_LOWER_PERCENTAGE = None

def inspect_dataset(dataset_path):
    path_type = sp.check_path(dataset_path)

    if path_type == "project":
        dataset = sp.parse_project_for_training(dataset_path)
        class_description = sp.extract_class_color_description(dataset_path)
        printi("Found " + str(len(dataset)) + " images in project")
    elif path_type == "dataset":
        dataset = sp.parse_dataset_for_training(dataset_path)
        # get path from dataset_path one level up
        class_description_path = os.path.dirname(dataset_path)
        class_description = sp.extract_class_color_description(class_description_path)
        printi("Found " + str(len(dataset)) + " images in dataset")

    if len(dataset) > 3:
        # get the number of masks from the first image
        img, masks = sp.parse_image(dataset[0][0], dataset[0][1],class_description,None)
        n_masks = len(masks)
    else:
        printe("Dataset does not contain enough images")
        return
    
    if SHOW and len(dataset) > 3:
        printi("Showing first 4 images with all masks")
        # show a plot with the image and all masks in subplots
        fig, ax = plt.subplots(4, n_masks + 1)
        for i in range(4):
            img, masks = sp.parse_image(dataset[i][0], dataset[i][1], class_description,None)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # remove axis
            ax[i, 0].axis('off')
            ax[i, 0].imshow(img)
            img_name = os.path.basename(dataset[i][0])
            ax[i, 0].set_title(img_name)
            j = 1
            for class_name, mask in masks.items():
                # remove axis
                ax[i, j].axis('off')
                ax[i, j].imshow(mask, cmap="gray")
                ax[i, j].set_title(class_name)
                j += 1
        #make fig bigger
        fig.set_size_inches(18.5, 10.5)
        plt.show()

    if PLOT:
        printi("Plotting pixel distribution on classes")
        plotter.plot_pixel_class_distribution(dataset, class_description)

    if len(dataset) > 32 and DG:
        printi("Testing DataGenerator with batch of 32")
        # test DataGenerator
        image_size = sp.get_image_size(dataset[0][0])

        resizing = (RESIZE_HEIGHT, RESIZE_WIDTH)
        if resizing == (None, None):
            resizing = None
        elif resizing[0] is None:
            resizing = (image_size[0], resizing[1])
        elif resizing[1] is None:
            resizing = (resizing[0], image_size[1])

        data_gen = dg.DataGenerator(dataset, image_size, n_masks, class_description, 32, resizing, USE_LOWER_PERCENTAGE)
        x, y = data_gen.__getitem__(0)
        printi(f"Shape of x: {x.shape}")
        printi(f"Shape of y: {y.shape}")
        printi(f"Number of batches: {len(data_gen)}")

        printi("Plotting first 4 images from DataGenerator")
        # show a plot with the image and all masks in subplots
        fig, ax = plt.subplots(4, n_masks + 2)
        for i in range(4):
            # remove axis
            ax[i, 0].axis('off')
            img = x[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i, 0].imshow(img)
            img_name = os.path.basename(dataset[i][0])
            ax[i, 0].set_title(img_name)
            for j in range(n_masks + 1):
                # remove axis
                ax[i, j+1].axis('off')
                ax[i, j+1].imshow(y[i,:,:,j], cmap="gray")
                ax[i, j+1].set_title(f"Class {j}")
        #make fig bigger
        fig.set_size_inches(18.5, 10.5)
        plt.show()



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Inspect a supervisely dataset')

    argparser.add_argument('dataset_path', type=str, help='Path to one dataset or the complete project')
    argparser.add_argument('-s', '--show', action='store_true', help='Show images', default=SHOW)
    argparser.add_argument('-p', '--plot', action='store_true', help='Plot statistics', default=PLOT)
    argparser.add_argument('-d', '--data-generator', action='store_true', help='Test Data Generator', default=PLOT)

    argparser.add_argument('--use-lower-percentage', type=float, help="Use only this percentage of the lower part of the image for training. This is useful if you want to train a model that only detects the lane in front of the car. Note: This will be done before a possible resizing, so this may change the 'original' resolution.", default=USE_LOWER_PERCENTAGE)
    argparser.add_argument('--resize-width', type=int, help='Resize image width to this size. If not set, the original size will be used but this can throw error if image sizes do not match between datasets.', default=RESIZE_WIDTH)
    argparser.add_argument('--resize-height', type=int, help='Resize image height to this size. Same as for resize-width.', default=RESIZE_HEIGHT)

    args = argparser.parse_args()

    SHOW = args.show
    PLOT = args.plot
    DG = args.data_generator
    USE_LOWER_PERCENTAGE = args.use_lower_percentage
    RESIZE_WIDTH = args.resize_width
    RESIZE_HEIGHT = args.resize_height

    inspect_dataset(args.dataset_path)