from lanedetection.models.unet import VGG16U, VGG8U
import torch
import torch.nn as nn
import sys
import coremltools as ct


RESIZE_WIDTH = 320
RESIZE_HEIGHT = 224
USE_LOWER_PERCENTAGE = 1.0

USE_BACKGROUND = True

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def exportCoreML(model):
    model.eval()
    example_input = torch.rand(1, 3, RESIZE_WIDTH, RESIZE_HEIGHT)
    traced_script_module = torch.jit.trace(model, example_input)

    coreml = ct.convert(traced_script_module, convert_to="mlprogram", inputs=[ct.TensorType(shape=example_input.shape)])
    coreml.save(f"./output/{model.__class__.__name__}.mlpackage")

if __name__ == "__main__":
    model = VGG8U(n_classes=7 if USE_BACKGROUND else 6)
    weights = sys.argv[1] if len(sys.argv) > 1 else None
    if weights:
        model.load_state_dict(torch.load(weights, map_location=device))
    else:
        model.load_pretrained(device)
    exportCoreML(model)