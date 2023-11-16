#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import json
import yaml

import include.supervisely_parser as sp
from include.utils import dotdict
from include.logger import printe, printw, printi
from include.data_generator import DataGenerator
import include.loss_functions as lf
import include.metrics as m
import include.plotter as plotter
from models.unet import UNet

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict, get_parameters
from tinygrad.ops import Device
from tinygrad.jit import TinyJit

VERBOSE = False

default_config = dotdict({
    "train_size": 0.8,
    "model": "vgg",
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "loss_function": "categorical_crossentropy",
    "name": "some_model",
    "resize_width": None,
    "resize_height": None,
    "use_lower_percentage": None
})

def run(dataset_path, output_path, params):
    # check if output directory exists
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    path_type = sp.check_path(dataset_path)
    printi(f"You provided a {path_type} as dataset path")
    if path_type == "project":
        # load flat array of image and mask paths
        dataset = sp.parse_project_for_training(dataset_path)
        class_description = sp.extract_class_color_description(dataset_path)
        printi("Found " + str(len(dataset)) + " images in project")
    elif path_type == "dataset":
        # load flat array of image and mask paths
        dataset = sp.parse_dataset_for_training(dataset_path)
        # get path from dataset_path one level up
        class_description_path = os.path.dirname(dataset_path)
        class_description = sp.extract_class_color_description(class_description_path)
        printi("Found " + str(len(dataset)) + " images in dataset")

    # sample dataset into train and validation set
    train_size = int(len(dataset) * params.train_size)
    train_set = dataset[:train_size]
    validation_set = dataset[train_size:]
    printi(f"Training set size: {len(train_set)}")
    printi(f"Validation set size: {len(validation_set)}")

    # load first image to get some info
    first_img, first_masks = sp.parse_image(train_set[0][0], train_set[0][1], class_description, 'background')
    image_height, image_width = first_img.shape[:2]
    number_classes = len(first_masks)

    # calculate sizes after possible resizing
    resizing = (params.resize_height, params.resize_width)
    if resizing == (None, None):
        resizing = None
    elif resizing[0] is None:
        resizing = (image_height, resizing[1])
    elif resizing[1] is None:
        resizing = (resizing[0], image_width)

    # create data generators
    data_gen_train = DataGenerator(train_set, (image_height, image_width), number_classes, class_description, params.batch_size, resizing, params.use_lower_percentage)
    data_gen_valid = DataGenerator(validation_set, (image_height, image_width), number_classes, class_description, params.batch_size, resizing, params.use_lower_percentage)

    # define input sizes for network
    if resizing is not None:
        input_height, input_width = resizing
    else:
        input_height, input_width = image_height, image_width

    # load loss function
    if params.loss_function == 'categorical_crossentropy':
        loss = lf.cross_entropy
    elif params.loss_function == 'dice':
        loss = lf.dice_loss
    elif params.loss_function == 'tversky':
        loss = lf.tversky_loss
    elif params.loss_function == 'focal_tversky':
        loss = lf.focal_tversky_loss
    else:
        printe(f"Unknown loss function {params.loss_function}")
        return
    
    # load metrics
    metrics = [m.iou_score, m.precision, m.recall, m.f1_score]

    # load network architecture
    # if params.model == 'vgg':
    #     model = vgg(params.name, input_height=input_height, input_width=input_width, number_classes=number_classes+1, learning_rate=params.learning_rate, loss_function=loss, metrics=metrics)
    # elif params.model == 'mobilenet':
    #     model = mobilenet(params.name, input_height=input_height, input_width=input_width, number_classes=number_classes+1, learning_rate=params.learning_rate, loss_function=loss, metrics=metrics)
    # elif params.model == 'tiny':from tinygrad.jit import TinyJit
    #     model = tiny(params.name, input_height=input_height, input_width=input_width, number_classes=number_classes+1, learning_rate=params.learning_rate, loss_function=loss, metrics=metrics)
    # else:
    #     printe(f"Unknown model {params.model}")
    #     return
    model = UNet(in_channels=3, n_class=number_classes)

    # load optimizer
    opt = Adam(get_parameters(model), lr=params.learning_rate)

    # check if directoy for model exists
    model_path = os.path.join(output_path, params.name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    printi(f"Starting training on {Device.DEFAULT}")

    @TinyJit
    def train_step_jitted(model, optimizer, X,Y):
         #forward pass
        out = model(X)
        # compute loss
        loss_tensor = loss(out, Y)
        # zero gradients
        optimizer.zero_grad()
        # backward pass
        loss_tensor.backward()
        # update weights
        optimizer.step()
        return loss_tensor.realize(), out.realize()

    # train model
    with Tensor.train():
        history = {v.__name__: [] for v in [loss] + metrics}
        for epoch in range(params.epochs):
            history_per_epoch = {v.__name__: [] for v in [loss] + metrics}
            for step in range(len(data_gen_train)):
                # sample a batch
                x, y = data_gen_train[step]
                batch = Tensor(x, requires_grad=False)
                batch = batch.permute(0, 3, 1, 2)
                labels = Tensor(y)
                labels = labels.permute(0, 3, 1, 2)
                # load into device (GPU)
                batch = batch.to(Device.DEFAULT)
                labels = labels.to(Device.DEFAULT)

                loss_tensor, out = train_step_jitted(model, opt, batch, labels)
                loss_cpu = loss_tensor.numpy()

                history_per_epoch[loss.__name__].append(loss_cpu)
                # calculate metrics
                for metric in metrics:
                    history_per_epoch[metric.__name__].append(metric(out, labels).realize().numpy())
            # calculate mean metrics
            for name in history_per_epoch.keys():
                history_per_epoch[name] = sum(history_per_epoch[name])/len(history_per_epoch[name])
                history[name].append(history_per_epoch[name])
            #print epoch status
            print(f"Epoch {epoch+1}/{params.epochs} - train: {history_per_epoch}")

    # save model
    model_file_path = os.path.join(model_path, "model.safetensor")
    state_dict = get_state_dict(model)
    safe_save(state_dict, model_file_path)
    printi(f"Saved model to {model_file_path}")
    
    # save training history
    history_file_path = os.path.join(model_path, "history.json")
    with open(history_file_path, 'w') as f:
        json.dump(history, f)
    printi(f"Saved training history to {history_file_path}")
    # plot training history
    if VERBOSE:
        show_plot = True
        printi("Showing training history plot. Close plot to continue.")
    else:
        show_plot = False
    history_plot_path = os.path.join(model_path, "history.png")
    plotter.plot_training_history(history, history_plot_path, show_plot=show_plot)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Train a lane detection model using a provided dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('dataset_path', type=str, help='Path to one dataset or the complete project')
    argparser.add_argument('-o', '--output', type=str, help='Path to the output directory', default='./output')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output', default=VERBOSE)

    # training parameters
    argparser.add_argument('-m', '--model', type=str, choices=['vgg', 'mobilenet', 'tiny'], help='Model to use for training', default=default_config.model)
    argparser.add_argument('-t', '--train-size', action="store", type=float, help=f"Percentage of dataset to use for training.", default=default_config.train_size)
    argparser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train', default=default_config.epochs)
    argparser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=default_config.batch_size)
    argparser.add_argument('-l', '--learning-rate', type=float, help='Learning rate', default=default_config.learning_rate)
    argparser.add_argument('-c', '--loss-function', type=str, choices=['categorical_crossentropy', 'dice', 'tversky', 'focal_tversky'], help='Loss function to use for training', default=default_config.loss_function)
    argparser.add_argument('-n', '--name', type=str, help='Name of the newly trained model', default=default_config.name)
    # image preprocessing parameters
    argparser.add_argument('--use-lower-percentage', type=float, help="Use only this percentage of the lower part of the image for training. This is useful if you want to train a model that only detects the lane in front of the car. Note: This will be done before a possible resizing, so this may change the 'original' resolution.", default=default_config.use_lower_percentage)
    argparser.add_argument('--resize-width', type=int, help='Resize image width to this size. If not set, the original size will be used but this can throw error if image sizes do not match between datasets.', default=default_config.resize_width)
    argparser.add_argument('--resize-height', type=int, help='Resize image height to this size. Same as for resize-width.', default=default_config.resize_height)

    argparser.add_argument('-y', '--yaml', type=str, help='Path to a YAML file containing configuration parameters', default=None)

    args = argparser.parse_args()
    VERBOSE = args.verbose

    params = dotdict(args.__dict__)

    if args.yaml is not None:
        with open(args.yaml, 'r') as f:
            yaml_params = yaml.safe_load(f)
        params.update(yaml_params)

    run(args.dataset_path, args.output, params)