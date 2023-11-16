from tinygrad.tensor import Tensor

def tp(out, y_true):
    y_true = y_true.permute(3,1,2,0)
    out = out.permute(3,1,2,0)
    y_true = y_true.flatten()
    out = out.flatten()
    
    return (y_true * out).sum()

def possible_positives(out, y_true):
    y_true = y_true.permute(3,1,2,0)
    y_true = y_true.flatten()
    return y_true.sum()

def predicted_positives(out, y_true):
    out = out.permute(3,1,2,0)
    out = out.flatten()
    return out.sum()

############

def fn(out, y_true):
    return possible_positives(out, y_true) - tp(out, y_true)

def fp(out, y_true):
    return predicted_positives(out, y_true) - tp(out, y_true)

def recall_(out, y_true):
    return tp(out, y_true) / (possible_positives(out, y_true) + 1e-5) # add epsilon for zero divison prevention

def recall(out, y_true):
    number_classes = out.shape[3]
    return 1/number_classes * recall_(out, y_true)

def precision_(out, y_true):
    return tp(out, y_true) / (predicted_positives(out, y_true) + 1e-5)

def precision(out, y_true):
    number_classes = out.shape[3]
    return 1/number_classes * precision_(out, y_true)

def f1_score(out, y_true):
    number_classes = out.shape[3]
    precision_m = precision_(out, y_true)
    recall_m = recall_(out, y_true)
    f1 =  2*((precision_m*recall_m)/(precision_m+recall_m+1e-5))
    return 1/number_classes * f1

def iou_score(out, y_true):
    number_classes = out.shape[3]
    precision_m = precision_(out, y_true)
    recall_m = recall_(out, y_true)
    iou = (precision_m * recall_m)/(precision_m + recall_m - precision_m * recall_m + 1e-5)
    return 1/number_classes * iou