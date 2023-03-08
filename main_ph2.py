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
import sys
from PIL import Image 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from imutils import paths
import numpy as np
import os
import warnings
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
import numpy
import tensorflow as tf

sys.path.insert(1, 'C:\\Users\\User\\ZZZ. ARE THEY BIOMARKERS\\Reproduction Code')

import data_loader_ph2
import model_maker_ph2

from data_loader_ph2 import load_CTCOV, load_CTX,load_Cells,load_MPIP,load_cmx, load_leukemia, load_petlidc, load_skin
from model_maker_ph2 import make_lvgg, make_xception, make_vgg
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import cv2

def gaussian (img_array):

    mean = 0
    vvv = 0.01
    sigma = vvv**0.5
    gaussian = np.random.normal(mean, sigma, (img_array.shape[0],img_array.shape[1]))
    
    noisy_image = np.zeros(img_array.shape, np.float32)
    noisy_image[:, :, 0] = img_array[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img_array[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img_array[:, :, 2] + gaussian
    
    
    
    return noisy_image


def train(epochs,batch_size, model, in_shape, tune, classes, n_split):

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
        
        if model == 'lvgg':
            model3 = make_lvgg(in_shape, tune, classes) #in every iteration we retrain the model from the start and not from where it stopped
        elif model == 'xception':
            model3 = make_xception(in_shape, tune, classes)
        elif model == 'vgg':
            model3 = make_vgg(in_shape, 20, classes)
       
        
        if omega == 1:
           model3.summary()
        omega = omega + 1   
        
        print('[INFO] PREPARING FOLD: '+str(omega-1))
        
        # model3.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
        
        # aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True) # AUG FOR NODULES
        
        aug = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range = 0.2, height_shift_range=0.2,rotation_range=180, horizontal_flip=True, vertical_flip=True)
        aug.fit(trainX)
        
        
        model3.fit_generator(aug.flow(trainX, trainY,batch_size=batch_size), epochs=epochs, steps_per_epoch=len(trainX)//batch_size)
        
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

    return model3, predictions_all,predictions_all_num,test_labels,labels, auc, conf_final,final_score,mean_f1




#%% TRAIN - OBTAIN RESULTS

  # GIVE PATH TO IMAGES
path = 'E:\\Data (Biomarkers phase 2)\\SKIN_NVLES\\'

  # LOAD IMAGES WITH DIFERRENT FUNCTIONS
#data, labels, labeltemp, image = load_CTCOV(path)
#data, labels, labeltemp, image = load_CTX(path)
#data, labels, labeltemp, image = load_petlidc(path)
data, labels, labeltemp, image = load_skin(path)



in_shape = (106,75,3) # must be same with image returned from load_[which data](path)
tune = 1 # SET: 1 FOR TRAINING SCRATCH, 0 FOR OFF THE SHELF, INTEGER FOR TRAINABLE LAYERS (FROM TUNE AND DOWN, THE LAYERS WILL BE TRAINABLE)
classes = 2
n_split=10 #10fold cross validation - PHASE 1
epochs = 20
batch_size = 16
modelo = 'xception' # PHASE 1



#%% FIT THE MODEL TO THE DATA (FOR PHASE 2)

model3 = make_lvgg(in_shape, 20, classes)
# model3 = make_xception(in_shape, 20, classes)
model3.summary()



aug = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range = 0.2, height_shift_range=0.2, rotation_range=180, horizontal_flip=True, vertical_flip=True)
#aug = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range = 0.2, height_shift_range=0.2,rotation_range=180, horizontal_flip=True, vertical_flip=True)
aug.fit(data)
        
model3.fit_generator(aug.flow(data, labels,batch_size=batch_size), epochs=epochs, steps_per_epoch=len(data)//batch_size)
        
model3.fit(data, labels, epochs=epochs, batch_size=batch_size)


# SAVE AND LOAD MODEL
model3.save('C:\\Users\\User\\ZZZ. ARE THEY BIOMARKERS\\Reproduction Code\\modelsaved.h5')

model = tf.keras.models.load_model('C:\\Users\\User\\ZZZ. ARE THEY BIOMARKERS\\Reproduction Code\\modelsaved.h5')

# tf.keras.utils.plot_model(
#     model,
#     to_file="C:\\Users\\User\\ZZZ. ARE THEY BIOMARKERS\\Reproduction Code\\model.png",
#     show_shapes=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
# )


#%%  TRAIN AND TEST ON THE SAME DATA (FOR PHASE 1)

# model3, predictions_all,predictions_all_num,test_labels,labels, auc, conf_final,final_score,mean_f1 = train(epochs,batch_size, modelo, in_shape, tune, classes, n_split)






#%%

  # GIVE PATH TO IMAGES
path2 = 'E:\\Data (Biomarkers phase 2)\\MNIST\\'

  # LOAD IMAGES WITH DIFERRENT FUNCTIONS
#data, labels, labeltemp, image = load_CTCOV(path)
#data, labels, labeltemp, image = load_CTX(path)
# data2, labels2, labeltemp2, image2 = load_petlidc(path)
data2, labels2, labeltemp2, image2 = load_skin(path2)

lab = labels2.argmax(axis=-1)
preds = model.predict(data2)
_,score2 = model.evaluate(data2,labels2)

auc2 = roc_auc_score(labels2, preds)

preds2 = preds.argmax(axis=-1)
conf_final = confusion_matrix(lab, preds2)
f1 = f1_score(lab, preds2)












