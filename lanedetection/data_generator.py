# Based on tensorflow/data_generator.py but without the tensorflow

import numpy as np
import lanedetection.supervisely_parser as sp
from lanedetection.utils import printe, getenv
import cv2
from typing import List, Tuple, Optional, Dict
        
class DataGenerator():
    def __init__(self, dataset: List[Tuple[str,str]], img_size: Tuple[int, int], number_classes: int, class_description: Dict[str, np.ndarray], batch_size: int, resizing: Optional[Tuple[int, int]], lower_percentage_crop: Optional[float], augmentation: bool = False, include_background: bool = False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.number_classes = number_classes
        self.class_description = class_description
        self.resizing = resizing
        self.lower_percentage_crop = lower_percentage_crop
        self.img_size = img_size
        self.augmentation = augmentation
        self.background_cls = 'background' if include_background else None
        self.seed = np.random.randint(0, 100000)

        self.img_size_after_preprocessing = self.img_size
        if lower_percentage_crop is not None:
            self.img_size_after_preprocessing = (int(self.img_size[0]*(1-self.lower_percentage_crop)), self.img_size[1])
        if resizing is not None:
            self.img_size_after_preprocessing = (self.resizing[0], self.resizing[1])

    @staticmethod
    def augment(img: np.ndarray, seed: int) -> np.ndarray:
        # Set the random seed
        np.random.seed(seed)
        grayscale = True if len(img.shape) == 2 else False
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
        rows, cols = img.shape[:2]
        if np.random.rand() < 0.5:
            crop_percentage = np.random.uniform(0.7, 1.0)
            x_offset = int((1 - crop_percentage) * cols / 2)
            y_offset = int((1 - crop_percentage) * rows / 2)
            img = img[y_offset:rows-y_offset, x_offset:cols-x_offset] if grayscale else img[y_offset:rows-y_offset, x_offset:cols-x_offset, :]
            img = cv2.resize(img, (cols, rows))
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1.0).astype(np.float32)
        return img

    @staticmethod
    def preprocess(img: np.ndarray, lower_percentage_crop: float, resizing: Tuple[int, int]) -> np.ndarray:
        height = img.shape[0]
        grayscale = True if len(img.shape) == 2 else False
        if lower_percentage_crop is not None:
            # crop the image to the lower_percentage_crop
            y_offset = int(height*(1-lower_percentage_crop))
            img = img[y_offset:,:] if grayscale else img[y_offset:,:,:]
            height = img.shape[0]
        # resize the image if set
        if resizing is not None:
            img = cv2.resize(img, (resizing[1], resizing[0]))
        return (img/255.0).astype(np.float32)
            
    def preprocess_(self,img: np.ndarray, seed: int) -> np.ndarray:
        img = DataGenerator.preprocess(img, self.lower_percentage_crop, self.resizing)
        if self.augmentation:
            return DataGenerator.augment(img, seed)
        return img
            
    def data_generation(self, dataset_batch: List[Tuple[str,str]]) -> Tuple[np.ndarray, np.ndarray]:
        # x : (n_samples, number_classes, *dim)
        # Initialization
        batch_size = len(dataset_batch)
        x = np.zeros((batch_size,) + (3,) + self.img_size_after_preprocessing, dtype=np.float32)
        y = np.zeros((batch_size,) + (self.number_classes,) + self.img_size_after_preprocessing, dtype=np.float32)
        # load data
        for i, (img_path, mask_path) in enumerate(dataset_batch):
            img, masks_dict = sp.parse_image(img_path, mask_path, self.class_description, self.background_cls)
            img = self.preprocess_(img, seed=self.seed)
            if img.shape[:2] != self.img_size_after_preprocessing:
                printe(f"Image {img_path} has wrong size. Expected {self.img_size_after_preprocessing} but got {img.shape[:2]}. Try to set resize_height and resize_width to {self.img_size}")
                return None, None
            x[i] = img.transpose(2,0,1)
            # sort masks_dict by key alphabetically
            #masks_dict = dict(sorted(masks_dict.items()))
            for j, mask in enumerate(masks_dict.values()):
                mask = self.preprocess_(mask, seed=self.seed)
                y[i,j,:,:] = mask
            self.seed += 1
        return x, y
    
    def __len__(self) -> int:
        return int(np.floor(len(self.dataset) / self.batch_size))
    
    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        batch_dataset = [self.dataset[k] for k in np.random.randint(0, len(self.dataset), self.batch_size)]
        return self.data_generation(batch_dataset)