import os, tqdm, random
import numpy as np
from typing import List, Tuple, Dict

import lanedetection.supervisely_parser as sp
from lanedetection.utils import printe, printw, printi, getenv
from lanedetection.data_generator import DataGenerator
from lanedetection.models.unet import VGG16U, VGG8U

from tinygrad import Tensor, nn, Device, TinyJit

# ********* hyperparameters *********

LEARNING_RATE = 1e-3
STEPS = 4000
BATCH_SIZE = 4
TRAIN_SIZE = 0.95
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 224
USE_LOWER_PERCENTAGE = 0.7
USE_BACKGROUND = False

PLOT = getenv("PLOT")

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/supervisely_project")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./output/")
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "model.safetensor")

def focal_tversky_loss(y_true: Tensor, y_pred: Tensor, smooth = 1e-5, alpha = 0.3, gamma=0.75):
    true_pos = (y_true * y_pred).sum(axis=(0,2,3)) # take sum over B, H, W
    false_neg = (y_true * (1-y_pred)).sum(axis=(0,2,3))
    false_pos = ((1-y_true) * y_pred).sum(axis=(0,2,3))
    return (1 - (true_pos) / (true_pos + (1-alpha)*false_neg + alpha*false_pos + smooth)).pow(gamma).mean()

def dice_loss(y_true: Tensor, y_pred: Tensor, smooth = 1e-5):
    intersection = (y_true * y_pred).sum(axis=(0,2,3))
    union = (y_true + y_pred).sum(axis=(0,2,3))
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()

def iou_score(y_true: Tensor, y_pred: Tensor, smooth = 1e-5) -> Tensor:
    intersection = (y_true * y_pred).sum(axis=(0,2,3))
    union = (y_true + y_pred).sum(axis=(0,2,3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()

def create_loss_graph(loss: List[float], val_loss: List[float]):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss, label="loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.savefig(os.path.join(OUTPUT_PATH, "loss.png"))

def create_iou_graph(ious: List[float], val_ious: List[float]):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ious, label="iou")
    plt.plot(val_ious, label="val_iou")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.savefig(os.path.join(OUTPUT_PATH, "iou.png"))

if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_PATH): os.mkdir(OUTPUT_PATH)

    path_type = sp.check_path(DATASET_PATH)
    class_description: Dict[str, np.ndarray] = sp.extract_class_color_description(os.path.dirname(DATASET_PATH) if path_type == "dataset" else DATASET_PATH)
    dataset: List[Tuple[str, str]] = sp.parse_dataset_for_training(DATASET_PATH) if path_type == "dataset" else sp.parse_project_for_training(DATASET_PATH)
    printi(f"Found {len(dataset)} images in {path_type}")

    random.shuffle(dataset)
    train_size = int(len(dataset) * TRAIN_SIZE)
    train_set, validation_set = dataset[:train_size], dataset[train_size:]
    printi(f"Training set size: {len(train_set)} | Validation set size: {len(validation_set)}")

    # load first image to get some info
    first_img, first_masks = sp.parse_image(train_set[0][0], train_set[0][1], class_description, 'background' if USE_BACKGROUND else None)
    image_height, image_width = first_img.shape[:2]
    n_classes = len(first_masks)
    resizing = (RESIZE_HEIGHT or image_height, RESIZE_WIDTH or image_width)
    data_gen_train = DataGenerator(train_set, (image_height, image_width), n_classes, class_description, BATCH_SIZE, resizing, USE_LOWER_PERCENTAGE, augmentation=True, include_background=USE_BACKGROUND)
    data_gen_valid = DataGenerator(validation_set, (image_height, image_width), n_classes, class_description, BATCH_SIZE, resizing, USE_LOWER_PERCENTAGE, include_background=USE_BACKGROUND)

    model = VGG8U(n_classes=n_classes)
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)

    loss_fn = dice_loss

    @TinyJit
    def train_step(x: Tensor, y: Tensor) -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(y, out)
            loss.backward()
            opt.step()
            return loss
    
    @TinyJit
    def get_masks(x: Tensor) -> Tensor:
        Tensor.no_grad = True
        out = model(x).realize()
        Tensor.no_grad = False
        return out

    window_sz = 20
    losses, val_losses, ious, val_ious = [float('inf') for _ in range(window_sz)], [float('inf')], [float('inf')], [float('inf')]
    printi(f"Starting training on {Device.DEFAULT} | model params: {sum([p.numel() for p in nn.state.get_parameters(model)]):,}")
    try:
        for step in (t:=tqdm.trange(STEPS)):
            x, y = data_gen_train()
            x,y = Tensor(x), Tensor(y)
            loss = train_step(x, y).item()
            losses.append(loss)

            if step % window_sz == 0 and step != 0:
                # validation
                val_x, val_y = data_gen_valid()
                val_x, val_y = Tensor(val_x), Tensor(val_y)
                val_y_pred = get_masks(val_x)
                val_losses.append(loss_fn(val_y, val_y_pred).item())
                # metric
                ious.append(iou_score(y, get_masks(x)).item())
                val_ious.append(iou_score(val_y, val_y_pred).item())
            t.set_description(f"loss: {sum(losses[-window_sz:])/window_sz:.5f} | val_loss: {val_losses[-1]:.5f} | iou_train: {ious[-1]:.3f} | iou_val: {val_ious[-1]:.3f}")
    except KeyboardInterrupt:
        printw("Training interrupted")
    # save model
    printi(f"Saving model to {OUTPUT_FILE}")
    state_dict = nn.state.get_state_dict(model)
    nn.state.safe_save(state_dict, OUTPUT_FILE)

    if PLOT: create_loss_graph([sum(losses[i:i+window_sz])/window_sz for i in range(0, len(losses), window_sz)], val_losses)
    if PLOT: create_iou_graph(ious, val_ious)
