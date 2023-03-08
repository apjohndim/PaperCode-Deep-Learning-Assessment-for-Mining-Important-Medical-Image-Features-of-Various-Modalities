import matplotlib as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time   # time1 = time.time(); print('Time taken: {:.1f} seconds'.format(time.time() - time1))
import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
import sys
from PIL import Image 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
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
import tensorflow as tf

sys.path.insert(1, 'C:\\Users\\User\\ZZZ. ARE THEY BIOMARKERS\\Reproduction Code')

#%%
import data_loader
import model_maker

from data_loader import load_CTCOV, load_CTX,load_Cells,load_MPIP,load_cmx, load_leukemia, load_petlidc, load_skin



  # GIVE PATH TO IMAGES
path = 'E:\\Data (Biomarkers)\\LIDC_NEW\\'

  # LOAD IMAGES WITH DIFERRENT FUNCTIONS
#data, labels, labeltemp, image = load_CTCOV(path)
data, labels, labeltemp, image = load_petlidc(path)



from model_maker import make_lvgg, make_densenet201, make_inceptionv3, make_mobilenetv2, make_nasnetmobile, make_resnet50, make_vgg,make_xception
#%% TRAIN - TEST - RESULTS PHASE
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

  # GLOBAL VARIABLES
in_shape = (80, 80,3) # must be same with image returned from load_[which data](path)
tune = 1 # SET: 1 FOR TRAINING SCRATCH, 0 FOR OFF THE SHELF, INTEGER FOR TRAINABLE LAYERS (FROM TUNE AND DOWN, THE LAYERS WILL BE TRAINABLE)
classes = 2
n_split=10 #10fold cross validation


scores = [] #here every fold accuracy will be kept
f1_scores = []
recalls = []
precisions = []
predictions_all = np.empty(0) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
conf_final = np.array([[0,0],[0,0]])
predictions_all_num = np.empty([0,classes])

omega = 1

# with tf.device('/cpu:0'):
    
for train_index,test_index in KFold(n_split).split(data):
    trainX,testX=data[train_index],data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    


    model3 = make_lvgg(in_shape, tune, classes) #in every iteration we retrain the model from the start and not from where it stopped
    if omega == 1:
       model3.summary()
    omega = omega + 1   
    
    print('[INFO] PREPARING FOLD: '+str(omega-1))
    
    model3.fit(trainX, trainY, epochs=10, batch_size=8)
    
    #aug = ImageDataGenerator(rotation_range=40, horizontal_flip=True, vertical_flip=True)
    #aug.fit(trainX)
    #model3.fit_generator(aug.flow(trainX, trainY,batch_size=8), epochs=20, steps_per_epoch=len(trainX)//)

    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    
    score = model3.evaluate(testX,testY)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    print ('[INFO] Accuracy ',score)
   
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array

    
    if classes == 2:
        recall = recall_score(testY2,predict)
        recalls.append(recall)
        
        precision = precision_score(testY2,predict)
        precisions.append(precision)
    
        oneclass = predict_num[:,1].reshape(-1,1)
        auc = roc_auc_score(testY2, oneclass)
    
        conf = confusion_matrix(testY2, predict) #get the fold conf matrix
        conf_final = conf + conf_final
    
        f1 = f1_score(testY2, predict)
        f1_scores.append(f1)
        
    

    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    testY = testY.argmax(axis=-1)
    test_labels = np.concatenate ([test_labels, testY]) #merge the two np arrays of labels


auc = roc_auc_score(labels, predictions_all_num)
rounded_labels=np.argmax(labels, axis=1)    
conf_final = confusion_matrix(rounded_labels, predictions_all)
    
scores = np.asarray(scores)
final_score = np.mean(scores)
f1sc = np.asarray(f1_scores)
mean_f1 = np.mean(f1sc)

