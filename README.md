 # Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Problem Statement: Handwritten Digit Recognition with Convolutional Neural Networks

Objective: Develop a Convolutional Neural Network (CNN) model to accurately classify handwritten digits (0-9) from the MNIST dataset.

Data: The MNIST dataset, a widely used benchmark for image classification, contains grayscale images of handwritten digits (28x28 pixels). Each image is labeled with the corresponding digit (0-9).

## Neural Network Model
![nueral network image](https://github.com/user-attachments/assets/87d136e3-925b-4fda-93bb-2c7937bcc1af)



## DESIGN STEPS

### STEP 1: Import libraries
### STEP 2: Load and preprocess data
### STEP 3: Define model architecture
### STEP 4: Compile the model
### STEP 5: Train the model
### STEP 6: Evaluate the model

## PROGRAM

### Name: Dario G
### Register Number: 212222230027
```py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape


X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')
print("Dario G")

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

 type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
print("Dario G ")

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

Name:Dario G

Register Number: 212222230027

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

print("Dario G")
metrics[['accuracy','val_accuracy']].plot()

print("Dario G")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print('\nDario G')

print('Dario G\n')
print(classification_report(y_test,x_test_predictions))

img = image.load_img('imagefive.jpeg')

type(img)

img = image.load_img('imagefive.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print('Dario G\n')
print(x_single_prediction)


plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print("Dario G\n")
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-09-23 103441](https://github.com/user-attachments/assets/d4b1e427-e3f9-487c-a35a-7c55d62d93b4)
![Screenshot 2024-09-23 103538](https://github.com/user-attachments/assets/7efeeda8-e61b-4c95-ad78-5773e768e4a6)



### Classification Report
![Screenshot 2024-09-23 103636](https://github.com/user-attachments/assets/a453f5dd-ccba-4351-a89d-c1fcf154a6eb)

### Confusion Matrix
![Screenshot 2024-09-23 103646](https://github.com/user-attachments/assets/0f371b36-e7e6-44bc-89c0-fdfbf7206a20)


### New Sample Data Prediction
![Screenshot 2024-09-23 103716](https://github.com/user-attachments/assets/5d8354b5-e7c1-4ba7-bc21-00bb954072d9)

![Screenshot 2024-09-23 103749](https://github.com/user-attachments/assets/1ca5a85a-fc5a-477f-988b-1b264506e543)



## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
