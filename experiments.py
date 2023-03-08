print("[INFO] Importing Libraries")
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
warnings.filterwarnings("ignore")
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from PIL import Image 
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras_applications.resnet import ResNet50
from keras_applications.mobilenet import MobileNet
SEED = 50   # set random seed
print("[INFO] Libraries Imported")


#adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#leakyrelu = keras.layers.LeakyReLU(alpha=0.3)
#elu = keras.layers.ELU(alpha=1.0)


input_img = Input(shape=(32, 32, 3)) 

from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.utils import plot_model
import tensorflow as tf



#%%   


#def make_model():
    
#import pydot
    
tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    
)


    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    # layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    # #base_model.summary()
    # for layer in base_model.layers:
    #     layer.trainable = False
    # for layer in base_model.layers[20:]:
    #     layer.trainable = True
    # #base_model.summary()
    
    # # early1 = layer_dict['block1_pool'].output
    # # #early1 = Conv2D(32, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early1)
    # # early1 = BatchNormalization()(early1)
    # # early1 = Dropout(0.5)(early1)
    # # early1= GlobalAveragePooling2D()(early1)
    # # #early1 = Flatten()(early1)
    
    # early2 = layer_dict['block2_pool'].output 
    # #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    # early2 = BatchNormalization()(early2)
    # early2 = Dropout(0.5)(early2)
    # early2= GlobalAveragePooling2D()(early2)
        
    # early3 = layer_dict['block3_pool'].output   
    # #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    # early3 = BatchNormalization()(early3)
    # early3 = Dropout(0.5)(early3)
    # early3= GlobalAveragePooling2D()(early3)    
        
    # early4 = layer_dict['block4_pool'].output   
    # #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    # early4 = BatchNormalization()(early4)
    # early4 = Dropout(0.5)(early4)
    # early4= GlobalAveragePooling2D()(early4)     
        
        
        
        
    # x1 = layer_dict['block5_conv3'].output 
    # x1= GlobalAveragePooling2D()(x1)
    # #x1 = Flatten()(x1)
    # x = keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    
    
    # #x = Flatten()(x)
    # #x = Dense(256, activation='relu')(x)
    
    # x = Dense(2500, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    
    # x = Dense(2, activation='softmax')(x)
    # model = Model(input=base_model.input, output=x)
    # #for layer in model.layers[:17]:
    #     #layer.trainable = True
    
     
    # # for layer in model.layers[17:]:
    # #     layer.trainable = True  
    # #model.summary()
    
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # #plot_model(model, to_file='vggmod19.png')
    # print("[INFO] Model Compiled!")
    #return model
  
#%%
  
print("[INFO] loading images from private data...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('C:\\Users\\User\\gan_classification_many datasets\\OLD DATASETS\\LIDC')))   # data folder with 2 categorical folders
random.seed(SEED)
random.shuffle(imagePaths)


# loop over the input images
for imagePath in imagePaths:
    # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), 
    # flatten the 32x32x3=3072 pixel image into a list, and store the image in the data list
    image = cv2.imread(imagePath)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))/255
    data.append(image)
 
    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float")
labeltemp=labels
labels = np.array(labels)

print("[INFO] Private data images loaded!")

print("Reshaping data!")

#data = data.reshape(data.shape[0], 32, 32, 1)

print("Data Reshaped to feed into models channels last")

print("Labels formatting")
lb = LabelBinarizer()
labels = lb.fit_transform(labels) 
labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
print("Labels ok!")

#%%
print("[INFO] loading images from dataset for test...")
data2 = []
labels2 = []


# grab the image paths and randomly shuffle them
imagePaths2 = sorted(list(paths.list_images('C:\\Users\\User\\gan_classification_many datasets\\self labeling')))   # data folder with 2 categorical folders
random.seed(SEED)
random.shuffle(imagePaths2)


# loop over the input images
for imagePath2 in imagePaths2:
    # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), 
    # flatten the 32x32x3=3072 pixel image into a list, and store the image in the data list
    image1 = cv2.imread(imagePath2)
    #image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (32, 32))/255
    data2.append(image1)
 
    # extract the class label from the image path and update the labels list
    label2 = imagePath2.split(os.path.sep)[-2]
    labels2.append(label2)


# scale the raw pixel intensities to the range [0, 1]
data2 = np.array(data2, dtype="float")
print("[INFO] Test data images loaded!")


lb = LabelBinarizer()
labels2 = lb.fit_transform(labels2) 
labels2 = keras.utils.to_categorical(labels2, num_classes=2, dtype='float32')
#data2 = data2.reshape(data2.shape[0], 32, 32, 1)

#%%
print("[INFO] loading images from dataset for test...")
data3 = []
labels3 = []


# grab the image paths and randomly shuffle them
imagePaths3 = sorted(list(paths.list_images('C:\\Users\\User\\gan_classification_many datasets\\PET')))   # data folder with 2 categorical folders
random.seed(SEED)
random.shuffle(imagePaths3)


# loop over the input images
for imagePath3 in imagePaths3:
    # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), 
    # flatten the 32x32x3=3072 pixel image into a list, and store the image in the data list
    image3 = cv2.imread(imagePath3)
    #image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image3 = cv2.resize(image3, (32, 32))/255
    data3.append(image3)
 
    # extract the class label from the image path and update the labels list
    label3 = imagePath3.split(os.path.sep)[-2]
    labels3.append(label3)


# scale the raw pixel intensities to the range [0, 1]
data3 = np.array(data3, dtype="float")
print("[INFO] Test data images loaded!")


lb = LabelBinarizer()
labels3 = lb.fit_transform(labels3) 
labels3 = keras.utils.to_categorical(labels3, num_classes=2, dtype='float32')
#data2 = data2.reshape(data2.shape[0], 32, 32, 1)


#%%
time1 = time.time() #initiate time counter
n_split=4 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces
conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
omega = 1
i = 0
j=4944

for train_index,test_index in KFold(n_split).split(data3):
    trainX,testX=data3[train_index],data3[test_index]
    trainY,testY=labels3[train_index],labels3[test_index]
    
    
    datastep = data2[i:j]
    labelstep = labels2 [i:j]
    i = int(i+ len(data2)*0.1)
    j = int(j+ len(data2)*0.1)
    trainX = np.concatenate([trainX,data, datastep])
    trainY =  np.concatenate([trainY, labels, labelstep])
    
    
    # trainX = np.concatenate([trainX, data3])
    # trainY =  np.concatenate([trainY, labels3])

    model3 = make_model() #in every iteration we retrain the model from the start and not from where it stopped
    if omega == 1:
       model3.summary()
    omega = omega + 1   
    
    print('[INFO] PREPARING FOLD: '+str(omega-1))
    #model3.fit(trainX, trainY,epochs=20, batch_size=64)
    
    aug = ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
    aug.fit(trainX)
    model3.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=1, steps_per_epoch=len(trainX)//64)
    score = model3.evaluate(testX,testY)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',model3.evaluate(testX,testY))
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    conf_final = conf + conf_final #sum it with the previous conf matrix
    name2 = name2 + 1
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
scores = np.asarray(scores)
final_score = np.mean(scores)


print("[INFO] Results Obtained!")
print('Time taken: {:.1f} seconds'.format(time.time() - time1)) 


#%%
#%% FOR THE PRETRAINED NET EVALUATION AND PREDICTION ON PET CT
scores = []


model3 = make_model()
model3.summary()
model3.fit(data, labels, epochs=4, batch_size=64)


print('Model evaluation ',model3.evaluate(data,labels))


#%%FOR THE PRETRAINED NET EVALUATION AND PREDICTION ON PET CT
labels2 = np.array(labels2)
labels2 = labels2.argmax(axis=-1)
prediction2 = model3.predict(data2)
prediction2 = prediction2.argmax(axis=-1)
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(labels2, prediction2))

print('Classification Report')
target_names = ['Benign', 'Malignant']
print(classification_report(labels2, prediction2, target_names=target_names))


#%%
    # BELOW IS A CODE TO SAVE THE INCORRECTLY CLASSIFIED INSTANCES 

prediction3 = model3.predict(data2)  
prediction3 = prediction3.argmax(axis=-1)
 


ori= data2[:,:,:,0]
    

for i in range (len(data2)): #for every image in testX
                im = Image.fromarray(ori[i]) #take the array and convert to image
                im2 = numpy.array(im, dtype=float)*255 # do it numpy array again an resize the pixel values
                im3 = Image.fromarray(im2) # convert back to image 
                im3 = im3.convert('RGB') # convert to 3channel
                name = i #give the img a unique name
                if prediction3[i] == 0:
                    name2 = 'pred_BENIGN'
                    name = 'C:\\Users\\User\\gan_classification_many datasets\\self labeling\\benign\\' + str(name2) +str(name)
                    im3.save(name + '.jpeg') #save to folder
                else: 
                    name2 = 'pred_MALIGNANT'
                    name = 'C:\\Users\\User\\gan_classification_many datasets\\self labeling\\malignant\\' + str(name2) +str(name)
                    im3.save(name + '.jpeg') #save to folder
                
              

