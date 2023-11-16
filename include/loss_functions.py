from tinygrad.tensor import Tensor

def categorial_crossentropy(out, y_true):
    return out.log_softmax().mul(y_true).mean()

def dice_loss(out, y_true, smooth=1e-5):
    out = out.permute(3,1,2,0)
    y_true = y_true.permute(3,1,2,0)

    out = out.flatten()
    y_true = y_true.flatten()
    intersection = (out * y_true).sum()
    dice = (2. * intersection + smooth) / (out.sum() + y_true.sum() + smooth)
    return 1 - dice

def tversky(out, y_true, smooth=1e-5, alpha=0.6):
   # out = out.permute(1,2,3,0)
    #y_true = y_true.permute(1,2,3,0)

    out = out.flatten()
    y_true = y_true.flatten()
    true_pos = out.mul(y_true).sum()
    false_neg = out.mul(1-y_true).sum()
    false_pos = (1-y_true).mul(out).sum()
    return (true_pos + alpha) / (true_pos + alpha*false_neg + alpha*false_pos + smooth)

def tversky_loss(out, y_true):
    return 1 - tversky(out, y_true)

def focal_tversky_loss(out, y_true, smooth = 1e-5, alpha = 0.5, gamma=0.75):
    tv = tversky_loss(out, y_true).pow(gamma)
    return (1-tv)**(gamma)

def cross_entropy(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
    y = (1 - label_smoothing)*y + label_smoothing / y.shape[1]
    if reduction=='none': return -x.log_softmax(axis=1).mul(y).sum(axis=1)
    if reduction=='sum': return -x.log_softmax(axis=1).mul(y).sum(axis=1).sum()
    return -x.log_softmax(axis=1).mul(y).sum(axis=1).mean()