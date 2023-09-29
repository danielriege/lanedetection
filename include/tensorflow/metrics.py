from keras import backend as K
import tensorflow as tf

# tensor like y_true = (B x H x W x C)
number_classes = 0

def tp(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),1)

def possible_positives(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_true = K.batch_flatten(y_true)
    return K.sum(K.round(K.clip(y_true, 0, 1)),1)

def predicted_positives(y_true, y_pred):
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
    y_pred = K.batch_flatten(y_pred)
    return K.sum(K.round(K.clip(y_pred, 0, 1)),1)

############

def fn(y_true, y_pred):
    return possible_positives(y_true, y_pred) - tp(y_true, y_pred)

def fp(y_true, y_pred):
    return predicted_positives(y_true, y_pred) - tp(y_true, y_pred)

def recall(y_true, y_pred):
    return tp(y_true, y_pred) / (possible_positives(y_true, y_pred) + K.epsilon()) # add epsilon for zero divison prevention

def recall_m(y_true, y_pred):
    return 1/number_classes * K.sum(recall(y_true, y_pred))

def precision(y_true, y_pred):
    return tp(y_true, y_pred) / (predicted_positives(y_true, y_pred) + K.epsilon())

def precision_m(y_true, y_pred):
    return 1/number_classes * K.sum(precision(y_true, y_pred))

def f1_score(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    f1 =  2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))
    return 1/number_classes * K.sum(f1)

def iou_score(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    iou = (precision_m * recall_m)/(precision_m + recall_m - precision_m * recall_m + K.epsilon())
    return 1/number_classes * K.sum(iou)