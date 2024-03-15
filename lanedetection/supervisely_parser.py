#!/usr/bin/env python3

import sys
import json
import os
import cv2
import numpy as np
from lanedetection.utils import printe, printi, printw
from typing import List, Tuple, Optional, Dict, Any

def extract_class_color_description(path_to_project: str) -> Optional[Dict[str, np.ndarray]]:
    """
    extract the class color description from the obj_class_to_machine_color.json file
    """
    obj_class_to_machine_colors = os.path.join(path_to_project, "obj_class_to_machine_color.json")
    if os.path.isfile(obj_class_to_machine_colors):
        with open(obj_class_to_machine_colors, 'r') as file:
            data = json.load(file)
        # change the color arrays into numpy arrays
        for class_name, color in data.items(): 
            data[class_name] = np.array(color)
        return data
    else:
        return None
    
def check_path(path: str) -> Optional[str]:
    """
    check if given path points to a project or dataset
    Returns:
    - "project" if path points to a project
    - "dataset" if path points to a dataset
    - None if path does not exist
    """
    if os.path.isdir(path):
        obj_class_file = extract_class_color_description(path)
        if obj_class_file is not None:
            return "project"
        else:
            return "dataset"
    elif os.path.isfile(path):
        if os.path.splitext(path)[1] == ".json":
            return "class_color_description"
        else:
            return "image"
    else:
        printe(f"Could not find {path}")
        return None

def check_project_structure(path_to_project: str) -> bool:
    """
    checks if the project structure is correct
    """
    # check if obj_class_to_machine_color.json exists
    if not os.path.isfile(os.path.join(path_to_project, "obj_class_to_machine_color.json")):
        printe("Could not find obj_class_to_machine_color.json. Please make sure your project contains this file.")
        return False
    # check if the project contains at least one dataset
    if len(os.listdir(path_to_project)) == 1:
        printe("Project does not contain any datasets.")
        return False
    # check if each dataset contains img and masks_machine directory
    for dataset in os.listdir(path_to_project):
        dataset_path = os.path.join(path_to_project, dataset)
        if os.path.isdir(dataset_path):
            if not os.path.isdir(os.path.join(dataset_path, "img")):
                printe(f"Dataset {dataset_path} does not contain img directory")
                return False
            if not os.path.isdir(os.path.join(dataset_path, "masks_machine")):
                printe(f"Dataset {dataset_path} does not contain masks_machine directory")
                return False
    return True

def parse_project_for_training(path_to_project: str) -> Optional[List[Tuple[str, str]]]:
    """
    parse a supervisely project will not load the images but only returns a flat array of image and corresponding mask paths.
    Later parse_image with these paths needs to be called to load the images
    """
    project = []
    for dataset_name in os.listdir(path_to_project):
        dataset_path = os.path.join(path_to_project, dataset_name)
        if os.path.isdir(dataset_path):
            dataset = parse_dataset_for_training(dataset_path)
            if dataset is not None:
                project += dataset
    return project

def parse_project(path_to_project: str, background_class:Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    parse a supervisely project
    masks will only be included if they contain at least one pixel (e.g. labled)
    returns a dictionary of datasets (key: dataset name, value: dictionary of images as in parse_dataset)
    """
    projects = {}
    class_color_description = extract_class_color_description(path_to_project)
    if class_color_description is None:
        printe("Could not extract class color description. Please make sure your project contains the file obj_class_to_machine_color.json.")
        return
    # list all directories in the project folder
    # for each directory, call parse_dataset
    for dataset in os.listdir(path_to_project):
        dataset_path = os.path.join(path_to_project, dataset)
        if os.path.isdir(dataset_path):
            parsed_dataset = parse_dataset(dataset_path, class_color_description, background_class)
            dataset_name = os.path.basename(dataset_path)
            projects[dataset_name] = parsed_dataset
            print("successfully parsed dataset " + dataset_name)
    return projects

def parse_dataset_for_training(path_to_dataset: str) -> Optional[List[Tuple[str, str]]]:
    """
    parse a supervisely dataset
    will not load the images but only returns a flat array of image and corresponding mask paths
    Later parse_image with these paths needs to be called to load the images
    """
    if os.path.isdir(path_to_dataset):
        dataset = []
        for img_file in os.listdir(os.path.join(path_to_dataset, "img")):
            img_file_path = os.path.join(path_to_dataset, "img", img_file)
            mask_file_path = os.path.join(path_to_dataset, "masks_machine", img_file)
            # change extension of mask_file_path to .png
            mask_file_path = os.path.splitext(mask_file_path)[0] + ".png"
            if is_labeled(img_file_path):
                dataset.append((img_file_path, mask_file_path))
        return dataset
    else:
        printe(f"Could not find dataset {path_to_dataset}")
        return

def parse_dataset(path_to_dataset: str, class_color_description: Dict[str, np.ndarray], background_class: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    parse a single dataset
    returns a dictionary of images (key: image name, value: dictionary of "img" and "masks") with masks being a dictionary of binary masks (one for each class: key: class name, value: binary mask)
    """
    dataset = {}
    #check if img and masks_machine directory is included in dataset
    if not os.path.isdir(os.path.join(path_to_dataset, "img")) and not os.path.isdir(os.path.join(path_to_dataset, "masks_machine")):
        printe(f"Dataset {path_to_dataset} does not contain img and masks_machine directory")
        return
    # load all images in img directory
    img_path = os.path.join(path_to_dataset, "img")
    img_files = os.listdir(img_path)
    # load all masks in masks_machine directory
    mask_path = os.path.join(path_to_dataset, "masks_machine")
    mask_files = os.listdir(mask_path)
    # check if number of images and masks is equal
    if len(img_files) != len(mask_files):
        printe(f"Number of images and masks is not equal in {path_to_dataset}")
        return
    # for each image, call parse_image
    for img_file in img_files:
        img_file_path = os.path.join(img_path, img_file)
        mask_file_path = os.path.join(mask_path, img_file)
        img, masks = parse_image(img_file_path, mask_file_path, class_color_description, background_class)
        img_name = os.path.splitext(img_file)[0]
        dataset[img_name] = {"img": img, "masks": masks}
    return dataset

def is_labeled(path_to_img: str) -> bool:
    """
    check if given image as path is labeled
    """

    # path to json is the same as to img, but instead of img dir it is in ann dir
    path_to_json = os.path.join(os.path.dirname(path_to_img).replace("img", "ann"), os.path.basename(path_to_img) + ".json")
    if os.path.isfile(path_to_json):
        with open(path_to_json, 'r') as file:
            data = json.load(file)
        if len(data["objects"]) > 0:
            return True
    return False

def parse_image(path_to_img: str, path_to_mask: str, class_color_description: Dict[str, np.ndarray], background_class: Optional[str]) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    parse a single image and mask
    returns the image and a dictionary of binary masks (one for each class)
    if background_class is set, a mask for the background will be included
    """
    masks = {}
    # load image and mask
    img = cv2.imread(path_to_img)
    mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
    # check if image and mask have same size
    if img.shape[:2] != mask.shape[:2]:
        printe(f"Image and mask have different size in {path_to_img}")
        return None
    # seperate mask into single binary masks (per color in class_color_description one class)
    number_of_found_classes = 0
    for class_name, color in class_color_description.items():
        color = np.array(color[0])
        binary_mask = cv2.inRange(mask, color, color)
        ###############################
        # SPECIAL RULE FOR OUTER LINE
        ###############################
        if "outer" in class_name:
            class_name = "outer"
        # only add mask if it contains at least one pixel
        if np.count_nonzero(binary_mask) > 0 or number_of_found_classes > 0:
            number_of_found_classes += 1
            if class_name in masks:
                # there is already a mask for this class
                # add the new mask to the existing one
                masks[class_name] = cv2.bitwise_or(masks[class_name], binary_mask)
            else:
                # there is no mask for this class yet
                masks[class_name] = binary_mask
    # optional create a mask for background
    if background_class is not None and number_of_found_classes > 0:
        background_mask = np.zeros(mask.shape, dtype=np.uint8)
        # create image where pixels are 255 if they are 0 in mask and 0 if they are not 0 in mask
        background_mask = cv2.inRange(mask, 0, 0)
        # only add mask if it contains at least one pixel
        if np.count_nonzero(background_mask) > 0:
            masks[background_class] = background_mask
    return img, masks

def get_image_size(path_to_img: str) -> Optional[Tuple[int, int]]:
    # load json of corresponding image
    path_to_json = os.path.join(os.path.dirname(path_to_img).replace("img", "ann"), os.path.basename(path_to_img) + ".json")
    if os.path.isfile(path_to_json):
        with open(path_to_json, 'r') as file:
            data = json.load(file)
        # get height and width from json
        height = data["size"]["height"]
        width = data["size"]["width"]
        return height, width
    else:
        printe(f"Could not find json for {path_to_img}")
        return None
