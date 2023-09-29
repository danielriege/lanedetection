#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import json
import yaml

import include.supervisely_parser as sp
from include.utils import dotdict
from include.logger import printe, printw, printi

from include.tensorflow.models import vgg, mobilenet, tiny
import include.tensorflow.loss_functions as lf
import include.plotter as plotter
import include.tensorflow.metrics as m
import include.tensorflow.data_generator as dg
import include.tensorflow.runner as tf_runner

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
    data_gen_train = dg.DataGenerator(train_set, (image_height, image_width), number_classes, class_description, params.batch_size, resizing, params.use_lower_percentage)
    data_gen_valid = dg.DataGenerator(validation_set, (image_height, image_width), number_classes, class_description, params.batch_size, resizing, params.use_lower_percentage)

    # define input sizes for network
    if resizing is not None:
        input_height, input_width = resizing
    else:
        input_height, input_width = image_height, image_width

    # load loss function
    if params.loss_function == 'categorical_crossentropy':
        loss = lf.categorical_crossentropy
    elif params.loss_function == 'weighted_ce':
        loss = lf.weighted_ce
    elif params.loss_function == 'dice':
        loss = lf.dice
    elif params.loss_function == 'tversky':
        loss = lf.tversky
    elif params.loss_function == 'focal_tversky':
        loss = lf.focal_tversky
    else:
        printe(f"Unknown loss function {params.loss_function}")
        return
    
    # load metrics
    metrics = [m.iou_score, m.precision_m, m.recall_m, m.f1_score]

    # load network architecture
    if params.model == 'vgg':
        model = vgg(params.name, input_height=input_height, input_width=input_width, number_classes=number_classes+1, learning_rate=params.learning_rate, loss_function=loss, metrics=metrics)
    elif params.model == 'mobilenet':
        model = mobilenet(params.name, input_height=input_height, input_width=input_width, number_classes=number_classes+1, learning_rate=params.learning_rate, loss_function=loss, metrics=metrics)
    elif params.model == 'tiny':
        model = tiny(params.name, input_height=input_height, input_width=input_width, number_classes=number_classes+1, learning_rate=params.learning_rate, loss_function=loss, metrics=metrics)
    else:
        printe(f"Unknown model {params.model}")
        return
    
    if VERBOSE:
        printi("Model summary:")
        model.summary()

    # check if directoy for model exists
    model_path = os.path.join(output_path, params.name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # train model
    history = tf_runner.train_model(model, data_gen_train, data_gen_valid, output_path, params, model_path)

    # save model
    model_file_path = os.path.join(model_path, "model.h5")
    model.save(model_file_path)
    printi(f"Saved model to {model_file_path}")
    
    # save training history
    history_file_path = os.path.join(model_path, "history.json")
    with open(history_file_path, 'w') as f:
        json.dump(history.history, f)
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
    argparser.add_argument('-c', '--loss-function', type=str, choices=['categorical_crossentropy', 'weighted_ce', 'dice', 'tversky', 'focal_tversky'], help='Loss function to use for training', default=default_config.loss_function)
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