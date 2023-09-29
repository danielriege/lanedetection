from keras import backend as K
import tensorflow as tf
import numpy as np

def dice(y_true, y_pred, smooth=1e-5):
    # make y_true per class
    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    # flatten per class
    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    
    intersection = K.sum(y_true_pos * y_pred_pos, 1)
    return (2 * intersection + smooth) / (K.sum(y_true_pos,1) +  K.sum(y_pred_pos,1) + smooth)

def dice_loss(y_true, y_pred):
    return K.sum(1 - dice(y_true, y_pred))

def tversky(y_true, y_pred, smooth=1e-5, alpha=0.6):

    # make y_true per class
    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    # flatten per class
    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return K.sum(1 - tversky(y_true, y_pred))


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.sum(K.pow((1 - tv), gamma))

def categorical_crossentropy(y_true, y_pred):    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    cross_entropy = K.log(y_pred + 1e-5) * y_true
    return - K.sum(cross_entropy)

def weighted_ce(y_true, y_pred):
    # weights for classes. Please check order of classes in y_pred
    # weights are aprox to pixel distribution
    #we = K.constant(np.array([[48], [73], [149], [185], [1250], [1110], [1]])) 
    we = K.constant(np.array([[2.07], [1.36], [0.67], [0.54], [0.08], [0.09], [100]])) 
    we = 1 - (we / K.sum(we))
    
    # outer, middle_curb, guide_lane, solid_lane, hold_line, zebra, background
    
    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    
    wce_per_class = we * y_true * K.log(y_pred + 1e-5)
    return - K.sum(wce_per_class)
     
    
    