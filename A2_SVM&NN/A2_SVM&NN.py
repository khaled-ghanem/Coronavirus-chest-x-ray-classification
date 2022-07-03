

import numpy as np 
import os
import cv2
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from PIL import Image

true_output=[]
Images=[]
labels = ['Negative', 'Positive']
sub_path="E:\Dataset" 
for label in labels: 
    path = os.path.join(sub_path, label)
    for img in os.listdir(path):
        img= os.path.join(path,img)
        im_gray = np.array(Image.open(img).convert('L'))
        #thresh = 500 # the images is black so we have to decrease the threshold value accuracy is too low (0.66)
        #thresh = 20 # the images is almost weight so we have to increase the threshold value and accuracy is too low (0.54)
        thresh = 110# Suitable value the accuracy is (1.0) which is perfect
        maxval = 255
        im_bin = (im_gray > thresh) * maxval
        Image.fromarray(np.uint8(im_bin)).save(img)#Binarize the input dataset with suitable threshold values
        Images.append(im_gray)#load data
        if(label=='Negative'):#create labels array
            true_output.append(0)
        else:
            true_output.append(1)
        #new_array = cv2.resize(im_gray, (90, 90))  #Resize images into suitable size
        
# Separate data into training and testing dataset
Images=np.array(Images)
print('Shape of data is :',Images.shape)
train_data= Images[:30]
d2_train_data=train_data.reshape(30,100*100)
test_data=Images[30:]
d2_test_data=test_data.reshape(10,100*100)

true_output=np.array(true_output)
train_true_output=true_output[:30]
test_true_output=true_output[30:]
print('Shape of labels is :',true_output.shape)
#____________________IMPLEMENTATION OF NEURAL NETWORK MODEL___________________________
# define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100)),
    #tf.keras.layers.Dense(140, activation='sigmoid'),# if we use this layer the accuracy get lower it will be(0.1) which is too low 
    tf.keras.layers.Dense(140, activation='relu'), # this number of hidden neurons (140) is suitable
    #tf.keras.layers.Dense(500, activation='relu'), #if we increse the number of hidden neurons(500) the accuracy get lower
    tf.keras.layers.Dense(2)# two output
])

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# fit the model
model.fit(train_data,train_true_output, epochs=10)
# evaluate the model
error, accuracy = model.evaluate(test_data,  test_true_output, verbose=2)
print('Accuracy using neural network:', accuracy)
print('Error using neural network:', error)
model.summary()
# make a prediction
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])# we use softmax on the output to predect the output for aparticular ratio
predictions = probability_model(test_data)
print('Predection using neural network:',labels[np.argmax(predictions[0])])

#____________________IMPLEMENTATION OF SVM MODEL___________________________
# define the model
#model =SVC(kernel='linear')# the accuracy if we used lineear kernel function is ZERO
#model =SVC(kernel='sigmoid')# the accuracy if we used rbf kernel function is ZERO
model =SVC(kernel='poly')# the accuracy if we used poly kernel function is (1.0)
# fit the model
model.fit(d2_train_data,train_true_output)
# evaluate the model
accuracy=model.score(d2_test_data,test_true_output)
print('Accuracy using SVM:',accuracy)
# make a prediction
predection=model.predict(d2_test_data)
print('Predection using SVM:',labels[predection[0]])



#Students names & IDs
#1- Khaled Mohamed Soliman Ghanem 20198027
#2- Ziad Ayman Mohamed 20198045
#3- Ahmed Ramadan Mohamed 20198116
#4- Amr Hosny Eid 20198059 

