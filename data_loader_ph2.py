## MODULE FOR IMPORTING ALL THE DATASETS
print("[INFO] Importing Libraries")
import matplotlib as plt
plt.style.use('ggplot')
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from PIL import Image 
import numpy
import keras
SEED = 50   # set random seed


#imgplot = plt.imshow(img)

def load_petlidc (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (80, 80))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image

def load_MPIP (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200,200))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image

def load_Cells (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    #labels = keras.utils.to_categorical(labels, num_classes=5, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image


def load_CTCOV (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (135, 135))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels, labeltemp, image

def load_CTX (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (250, 250))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    # labels = keras.utils.to_categorical(labels, num_classes=7, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image

def load_leukemia (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (75, 100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image


def load_skin (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 150))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image




def load_cmx (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    #labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image



def load_colon (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image


def load_lung (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    # labels = keras.utils.to_categorical(labels, num_classes=3, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image


def load_lymph (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    #labels = keras.utils.to_categorical(labels, num_classes=3, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image

def load_break (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (175, 115))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
        
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image


def load_cervical (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
        
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    # labels = keras.utils.to_categorical(labels, num_classes=3, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image

def load_prostate (path):    
    import os
    import pandas as pd 
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    
    filepath = path+'train_fixed.csv'
    
    file = pd.read_csv(filepath)
    file = np.array(file)
  
    labels = []
    
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        img_id = os.path.basename(imagePath)
        img_id = img_id[:-4]
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))/image.max()
        data.append(image)
        

        index = np.where(file== img_id)
        index = index[0][0]
        
        label = file[index,2]
        
        labels.append(label)
    data = np.array(data, dtype="float")
    
    labeltemp = 0
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 

    #labels = keras.utils.to_categorical(labels, num_classes=10, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image
