from typing import Union, List, Tuple
import os, cv2, time, tqdm, sys
import numpy as np

from lanedetection.data_generator import DataGenerator
from lanedetection.models.unet import VGG16U, VGG8U
import lanedetection.supervisely_parser as sp
from lanedetection.utils import printe, printw, printi, getenv

import torch
import torch.nn as nn

RESIZE_WIDTH = 320
RESIZE_HEIGHT = 224
USE_LOWER_PERCENTAGE = 1.0

USE_BACKGROUND = True

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_video(path: str) -> np.ndarray:
    """ Loads video into RAM """
    assert os.path.exists(path), f"File {path} does not exist."
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def load_dataset(datasets: List[Tuple[str,str]]) -> np.ndarray: 
    return np.array([cv2.imread(img_path) for img_path, _ in datasets])

def rgb_from_masks(masks: np.ndarray) -> np.ndarray:
    """ Converts masks to RGB """
    idx_to_color = {
        0: [0,0,1], # outer
        1: [0,1,0], # dashed
        2: [0,1,1], # solid
        3: [1,0,0], # hold line
        4: [1,0,1],  # area
        5: [1,1,0], # zebra
    }
    assert len(masks.shape) == 3, f"masks has wrong shape."
    rgb = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.float32)
    for j in range(3):
        for i in range(masks.shape[0]):
            masks = np.where(masks > 0.8, 1, 0)
            rgb[:,:,j] += masks[i,:,:] * idx_to_color[i][j]
    return rgb
    

def inference(input: str, model: object) -> None:
    """
    Performs the inference of the model on the input data.
    If input is more than 10 images, it will be displayes as video with 10 FPS.
    """
    assert os.path.exists(input), f"File {input} does not exist."
    ext = os.path.splitext(input)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']: data = np.array(cv2.imread(input))
    elif ext in ['.avi', '.mp4', '.mov', '.mkv']: data = load_video(input)
    elif ext in ['.npy']: data = np.load(input)
    elif sp.check_path(input) == "dataset": data = load_dataset(sp.parse_dataset_for_training(input))
    elif sp.check_path(input) == "project": data = load_dataset(sp.parse_project_for_training(input))
    assert data is not None, f"File {input} has wrong format."
    assert len(data.shape) == 4, f"data has wrong shape."

    model.to(device)
    model.eval()
    
    wait_time = 30 if data.shape[0] > 10 else 0
    times = []
    cv2.namedWindow("Inference", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    for step in (t:=tqdm.trange(data.shape[0])):
        st = time.perf_counter()
        x = torch.from_numpy(DataGenerator.preprocess(data[step], USE_LOWER_PERCENTAGE, (RESIZE_HEIGHT, RESIZE_WIDTH)).transpose(2,0,1)).to(device).unsqueeze(0)
        with torch.no_grad():
            y = model(x)[0].cpu().numpy()
        #cv2.imshow("test", cv2.cvtColor(y.transpose(1,2,0)[:,:,1].astype(np.float32), cv2.COLOR_GRAY2BGR))
        times.append((time.perf_counter() - st)*1000)
        cv2.imshow("Inference", rgb_from_masks(y[:-1,:,:])*255)
        cv2.imshow("Input", data[step])
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
        t.set_description(f"Step: {step+1}/{data.shape[0]} | {times[-1]:.2f} ms/step")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = VGG8U(n_classes=7 if USE_BACKGROUND else 6)
    weights = sys.argv[1] if len(sys.argv) > 2 else None
    data_input = sys.argv[2] if weights else sys.argv[1]
    if weights:
        model.load_state_dict(torch.load(weights, map_location=device))
    else:
        model.load_pretrained(device)
    inference(data_input, model)
        
        

