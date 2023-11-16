# Based on tensorflow/data_generator.py but without the tensorflow

import numpy as np
import include.supervisely_parser as sp
import include.logger as logger
import cv2
        
class DataGenerator():
    def __init__(self, dataset, img_size, number_classes, class_description, batch_size, resizing, lower_percentage_crop, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.number_classes = number_classes
        self.class_description = class_description
        self.resizing = resizing
        self.lower_percentage_crop = lower_percentage_crop
        self.img_size = img_size

        self.img_size_after_preprocessing = self.img_size
        if lower_percentage_crop is not None:
            self.img_size_after_preprocessing = (int(self.img_size[0]*(1-self.lower_percentage_crop)), self.img_size[1])
        if resizing is not None:
            self.img_size_after_preprocessing = (self.resizing[0], self.resizing[1])

        self.on_epoch_end()
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def preprocess(self,img):
        # check if image is grayscale
        if len(img.shape) == 2:
            # preprocessing a mask
            height, width = img.shape
            grayscale = True
        else:
            height, width, _ = img.shape
            grayscale = False
        # if we have lower_percentage_crop, crop the img
        if self.lower_percentage_crop is not None:
            # crop the image to the lower_percentage_crop
            y_offset = int(height*(1-self.lower_percentage_crop))
            if grayscale:
                img = img[y_offset:,:]
            else:
                img = img[y_offset:,:,:]
            height = img.shape[0]
        # resize the image if set
        if self.resizing is not None:
            img = cv2.resize(img, (self.resizing[1], self.resizing[0]))
        # normalize image
        img = img.astype(np.float32)/255.0
        return img

            
    def data_generation(self, dataset_batch):
        # x : (n_samples, *dim, number_classes)
        # Initialization
        batch_size = len(dataset_batch)
        x = np.zeros((batch_size,) + self.img_size_after_preprocessing + (3,), dtype="float32")
        y = np.zeros((batch_size,) + self.img_size_after_preprocessing + (self.number_classes,), dtype="float32")
        
        # load data
        for i, (img_path, mask_path) in enumerate(dataset_batch):
            img, masks_dict = sp.parse_image(img_path, mask_path, self.class_description, 'background')
            img = self.preprocess(img)
            if img.shape[:2] != self.img_size_after_preprocessing:
                logger.printe(f"Image {img_path} has wrong size. Expected {self.img_size_after_preprocessing} but got {img.shape[:2]}. Try to set resize_height and resize_width to {self.img_size}")
                return None, None
            x[i] = img
            # sort masks_dict by key alphabetically
            masks_dict = dict(sorted(masks_dict.items()))
            for j, mask in enumerate(masks_dict.values()):
                # add mask to y
                mask = self.preprocess(mask)
                y[i,:,:,j] = mask
            
            # if self.augmentation and self.transform is not None:
            #     transformed = self.transform(image=img, mask=mask)
            #     img = transformed['image']
            #     mask = transformed['mask']
        
        #print(f"whole batch took: {(time.time()-start_b)*1000}ms")
        return x, y
    
    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_dataset = [self.dataset[k] for k in indexes]

        # Generate data
        x, y = self.data_generation(batch_dataset)

        return x, y