# train landetection (segmentation) model

Pipeline to train a simple unet segmentation model to detect lane markings. Tinygrad is used as tensor lib.

use `python3 train.py` to start training. Dataset is not open source yet, but will be released soon.

### Env vars
for train.py:
- `PLOT=1` Save plots of the training process

for inspect_dataset.py: