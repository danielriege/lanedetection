from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D, Concatenate, AveragePooling1D, Reshape, Activation, add, Conv2DTranspose, BatchNormalization, UpSampling2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model, Sequence
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, BinaryCrossentropy
from tensorflow.keras.layers import Lambda
from tensorflow.keras.activations import softmax
from tensorflow import roll, norm, keras
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.applications as A

from . import metrics as m

'''
VGG16
'''
def vgg(name, input_height, input_width, number_classes, learning_rate, loss_function, metrics = None):
    m.number_classes = number_classes
    # base_model = A.ResNet50V2(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3)) 
    #base_model = A.InceptionV3(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3)) 
    base_model = A.VGG16(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3))
    # get end of blocks for skip connections
    block5_output = base_model.get_layer('block5_conv3').output
    block4_output = base_model.get_layer('block4_conv3').output
    block3_output = base_model.get_layer('block3_conv3').output
    block2_output = base_model.get_layer('block2_conv2').output
    block1_output = base_model.get_layer('block1_conv2').output
    blocks = [block5_output, block4_output, block3_output, block2_output, block1_output]

    pool = Conv2D(1024, 3, strides=1, padding='same')(base_model.output)
    x = Activation("relu")(pool)
    
    # decoder
    filters = [512,256,128,64,16]
    for i,f in enumerate(filters):
        x = Conv2D(f, 3, strides=1, padding='same')(x)
        x = UpSampling2D(interpolation='bilinear')(x)
        x = Activation("relu")(x)
        
        x = Concatenate()([blocks[i], x])

        x = Conv2D(f, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)
        x = Conv2D(f, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)
    
    output = Conv2D(number_classes, 1)(x)
    output = Activation('softmax')(output)
    
    model = Model(inputs=base_model.inputs, outputs=output, name=name)
    optimizer = Adam(lr=learning_rate) # lr is learning rate 3e-4
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics) # mean squared error because it is a regression problem
    #plot_model(model, to_file='%s.png' % (name))
    return model

'''
MobileNetV2
'''
def mobilenet(name, input_height, input_width, number_classes, learning_rate, loss_function, metrics = None):
    m.number_classes = number_classes
    base_model = A.MobileNetV2(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3))
    
    skip_connection_names = ["input","block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = base_model.get_layer("block_13_expand_relu").output
    
    filters = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        if skip_connection_names[-i] == "input":
            x_skip = base_model.input
        else:
            x_skip = base_model.get_layer(skip_connection_names[-i]).output
        
        x = Conv2D(filters[-i], 3, strides=1, padding='same')(x)
        x = UpSampling2D(2, interpolation='bilinear')(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(filters[-i], 3, strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)
        
        x = Conv2D(filters[-i], 3, strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)
        
    x = Conv2D(number_classes, 1, padding="same")(x)
    output = Activation("softmax")(x)
    
    model = Model(inputs=base_model.inputs, outputs=output, name=name)
    optimizer = Adam(lr=learning_rate) # lr is learning rate
    # tversky = 7e-4
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics) # mean squared error because it is a regression problem
    #plot_model(model, to_file='%s.png' % (name))
    return model

'''
Tiny 
'''
def tiny(name, input_height, input_width, number_classes, learning_rate, loss_function, metrics = None):
    m.number_classes = number_classes
    base_model = A.MobileNetV2(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3))
    
    skip_connection_names = ["input","block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = base_model.get_layer("block_13_expand_relu").output
    
    filters = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        if skip_connection_names[-i] == "input":
            x_skip = base_model.input
        else:
            x_skip = base_model.get_layer(skip_connection_names[-i]).output
        
        x = UpSampling2D(2, interpolation='bilinear')(x)
        x = Concatenate()([x, x_skip])
        
        x = DepthwiseConv2D(3, padding="same")(x)
        x = Conv2D(filters[-i], 1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)
        
        x = DepthwiseConv2D(3, padding="same")(x)
        x = Conv2D(filters[-i], 1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)
        
    x = Conv2D(number_classes, 1, padding="same")(x)
    output = Activation("softmax")(x)
    
    model = Model(inputs=base_model.inputs, outputs=output, name=name)
    optimizer = Adam(lr=learning_rate) # lr is learning rate
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics) # mean squared error because it is a regression problem
    #plot_model(model, to_file='%s.png' % (name))
    return model