import lanedetection.supervisely_parser as sp
from lanedetection.data_generator import DataGenerator
from lanedetection.utils import getenv, printe, printw, printi
from typing import List, Tuple, Dict, Optional

import os, cv2, tqdm
import matplotlib.pyplot as plt
import numpy as np

DG = getenv("DG") # test DataGenerator
DIST = getenv("DIST") # plot pixel class distribution
SHOW = getenv("SHOW") # show first 4 images

RESIZE_WIDTH = 160
RESIZE_HEIGHT = 64
USE_LOWER_PERCENTAGE = 0.7

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/supervisely_project/knuff_main1")

def plot_pixel_class_distribution(dataset, class_color_description):

    def calc_class_distribution_data(dataset, class_color_description) -> Tuple[Dict[str, int], Dict[str, int], int]:
        num_pixels_per_class, num_images_per_class, total_num_pixels = {}, {}, 0
        if not isinstance(dataset, list):
            printe("Dataset is not a list of path names")
            return None
        # show progress bar to 100%
        for img in tqdm.tqdm(dataset, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc="Calculating class distribution data"):
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

    hist_by_pixel, hist_by_images, total_num_pixels = calc_class_distribution_data(dataset, class_color_description)
    class_names = list(hist_by_pixel.keys())
    f, axs = plt.subplots(1,2,figsize=(30,5))
    x_pos = [i for i, _ in enumerate(class_names)]
    bar1 = axs[0].barh(x_pos, hist_by_pixel.values())
    axs[0].set_ylabel("Class")
    axs[0].set_xlabel("Num of Pixels")
    axs[0].set_yticks(x_pos)
    axs[0].set_yticklabels(class_names)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[0].text(0.7, 0.95, f'Total Pixel Number = {total_num_pixels:,.0f}', transform=axs[0].transAxes, fontsize=12, verticalalignment='top', bbox=props)
    for rect in bar1:
        width = rect.get_width()
        if width < 1_500_000:
            ha = 'left'
            pos = width + 10000
            color = 'black'
        else:
            ha = 'right'
            pos = width - 10000
            color = 'white'
        ratio = width / total_num_pixels
        axs[0].text(pos, rect.get_y() + rect.get_height() / 2.0, f'{width:,.0f} / {ratio*100:.2f}%', ha=ha, va='center', color=color, fontweight='bold')
    bar2 = axs[1].barh(x_pos, hist_by_images.values())
    axs[1].set_ylabel("Class")
    axs[1].set_xlabel("Number of Images")
    axs[1].set_yticks(x_pos)
    axs[1].set_yticklabels(list(hist_by_images.keys()))

    for rect in bar2:
        width = rect.get_width()
        if width < 500:
            ha = 'left'
            pos = width + 7
            color = 'black'
        else:
            ha = 'right'
            pos = width - 7
            color = 'white'
        axs[1].text(pos, rect.get_y() + rect.get_height() / 2.0, f'{width:,.0f}', ha=ha, va='center', color=color, fontweight='bold')
    plt.show()

if __name__ == "__main__":
    path_type = sp.check_path(DATASET_PATH)
    class_description: Dict[str, np.ndarray] = sp.extract_class_color_description(os.path.dirname(DATASET_PATH) if path_type == "dataset" else DATASET_PATH)
    dataset: List[Tuple[str, str]] = sp.parse_dataset_for_training(DATASET_PATH) if path_type == "dataset" else sp.parse_project_for_training(DATASET_PATH)
    printi(f"Found {len(dataset)} images in {path_type}")

    if len(dataset) > 3:
        # get the number of masks from the first image
        img, masks = sp.parse_image(dataset[0][0], dataset[0][1],class_description,None)
        n_classes = len(masks)
    else:
        printe("Dataset does not contain enough images")
        exit(1)
    
    if SHOW and len(dataset) > 3:
        printi("Showing first 4 images with all masks")
        # show a plot with the image and all masks in subplots
        fig, ax = plt.subplots(4, n_classes + 1)
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

    if DIST:
        printi("Plotting pixel distribution on classes")
        plot_pixel_class_distribution(dataset, class_description)

    if DG and len(dataset) > 32:
        printi("Testing DataGenerator with batch of 32")
        # test DataGenerator
        image_size = sp.get_image_size(dataset[0][0])
        resizing = (RESIZE_HEIGHT or image_size[0], RESIZE_WIDTH or image_size[1])
        data_gen = DataGenerator(dataset, image_size, n_classes+1, class_description, 32, resizing, USE_LOWER_PERCENTAGE)
        x, y = data_gen()
        printi(f"Shape of x: {x.shape}")
        printi(f"Shape of y: {y.shape}")
        printi(f"Number of batches: {len(data_gen)}")

        printi("Plotting first 4 images from DataGenerator")
        # show a plot with the image and all masks in subplots
        fig, ax = plt.subplots(4, n_classes + 2)
        for i in range(4):
            # remove axis
            ax[i, 0].axis('off')
            ax[i, 0].imshow(cv2.cvtColor(x[i].transpose(1,2,0), cv2.COLOR_BGR2RGB))
            ax[i, 0].set_title(os.path.basename(dataset[i][0]))
            for j in range(n_classes + 1):
                # remove axis
                ax[i, j+1].axis('off')
                ax[i, j+1].imshow(y[i,j,:,:], cmap="gray")
                ax[i, j+1].set_title(f"Class {j}")
        #make fig bigger
        fig.set_size_inches(18.5, 10.5)
        plt.show()