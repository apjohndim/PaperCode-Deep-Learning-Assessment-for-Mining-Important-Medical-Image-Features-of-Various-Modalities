
from imutils import paths
import numpy as np
import os
import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
import numpy
from keras_applications.resnet import ResNet50
from keras_applications.mobilenet import MobileNet


#adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#leakyrelu = keras.layers.LeakyReLU(alpha=0.3)
#elu = keras.layers.ELU(alpha=1.0)


from keras.applications.vgg19 import VGG19
from keras.utils import plot_model
import tensorflow as tf

#in_shape = (300, 300, 3)
#classes = 2

def make_lvgg (in_shape, tune, classes):
    
#import pydot
    
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[20:]:
        layer.trainable = True
    #base_model.summary()
    
    early2 = layer_dict['block2_pool'].output 
    #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    early2 = BatchNormalization()(early2)
    early2 = Dropout(0.5)(early2)
    early2= GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_pool'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = BatchNormalization()(early3)
    early3 = Dropout(0.5)(early3)
    early3= GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = BatchNormalization()(early4)
    early4 = Dropout(0.5)(early4)
    early4= GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model


#%%
def make_vgg (in_shape, tune, classes): #tune = 0 is off the self, tune = 1 is scratch, tune 
    
#import pydot
    
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    #base_model.summary()
  
    
    x1 = layer_dict['block5_conv3'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

#%%
def make_xception (in_shape, tune, classes):
    
    base_model = keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['block14_sepconv2'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 

#%%
def make_resnet50 (in_shape, tune, classes):
    
    base_model = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['res5c_branch2c'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x1)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

#%%
def make_inceptionv3 (in_shape, tune, classes):
    
    base_model = keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['mixed10'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
   # x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

#%%
def make_mobilenetv2 (in_shape, tune, classes):
    
    base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['out_relu'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x1)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

#%%
def make_densenet201 (in_shape, tune, classes):
    
    base_model = keras.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['relu'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x1)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

#%%
def make_nasnetmobile (in_shape, tune, classes):
    
    base_model = keras.applications.NASNetMobile(
    input_shape=in_shape,
    include_top=False,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['normal_concat_12'].output 
    x1= GlobalAveragePooling2D()(x1)

    x = Dense(2500, activation='relu')(x1)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model