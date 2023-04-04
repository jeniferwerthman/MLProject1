# ######################################################
# Code 5-14
# CIFAR-10 Recognition using CNN
# Dataset: CIFAR-10
# Machine Learning and Deep Learning Course
# Kent State University
# Jungyoon Kim, Ph.D.
# ######################################################
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import pandas as pd
import os
import tensorflow as tf
import cv2  
 
PATH = 'C:\Projects\\archive\\images'
train_dir = PATH + '\\train'
validation_dir = PATH + '\\validation'
BATCH_SIZE = 1                                              
IMG_SIZE = (32,32) 
no_of_classes = 7
# Read the CIFAR-10 dataset and convert it into a form to be input to the neural network
#(x_train,y_train),(x_test,y_test)=cifar10.load_data()
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE, 
                                                            image_size=IMG_SIZE)
training_Count = tf.data.experimental.cardinality(train_dataset).numpy()
#x_trainORG,y_trainORG = np.concatenate([x ,y for x, y in train_dataset], axis=0)
x_trainORG, y_trainORG = tuple(zip(*train_dataset))

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE, 
                                                                 image_size=IMG_SIZE)
class_names = train_dataset.class_names
validation_Count = tf.data.experimental.cardinality(validation_dataset).numpy()
#X_validORG,y_validORG = np.concatenate([x,y for x, y in train_dataset], axis=0)
X_validORG, y_validORG = tuple(zip(*train_dataset))

print('Number of training batches: %d' % training_Count)
print('Number of validation batches: %d' % validation_Count)


x_train, x_test, y_train, y_test = train_test_split( x_trainORG, y_trainORG, train_size=0.3)
x_Valtest, X_valid, y_Valtest, y_valid=train_test_split(X_validORG,y_validORG, train_size=.3)     
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_Valtest = np.array(x_Valtest)
X_valid = np.array(X_valid)
y_Valtest = np.array(y_Valtest)
y_valid = np.array(y_valid)
IMG_HEIGHT = 32
IMG_WIDTH = 32

def create_dataset(img_folder):
       
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name
# extract the image array and class name
x_train, y_train =create_dataset(train_dir)
x_train, x_test, y_train, y_test = train_test_split( x_train, y_train, train_size=0.3)
print("y_train", y_train)
#x_train=x_train.astype(np.float32)/255.0
#x_test=x_test.astype(np.float32)/255.0
##y_train=tf.keras.utils.to_categorical(y_train,10)
##y_test=tf.keras.utils.to_categorical(y_test,10)

# Neural network model design
cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(512,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10,activation='softmax'))

# Train neural network model
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# Evaluate neural network model accuracy
res=cnn.evaluate(x_test,y_test,verbose=0)
print("Accuracy",res[1]*100)

import matplotlib.pyplot as plt

# Accuracy graph
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()

# Loss function graph
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()
cnn.save("my_cnn.h5")