import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation
from keras import layers
from keras.utils import to_categorical
import matplotlib as plt
from matplotlib import pyplot as plt
import numpy as np
import os
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train.shape
x_test.shape

os.mkdir('dataset')
os.mkdir('dataset/train')
os.mkdir('dataset/test')

# 10=bowl 29=dinosaur 50=mouse 63=porcupine 90=train 91=trout
classNumber = [10, 29, 50, 63, 90, 91]

for num in classNumber:
    path=os.path.join('dataset/train', str(num))
    os.mkdir(path)
    path=os.path.join('dataset/test', str(num))
    os.mkdir(path)



for i in range(50000):
    if int(y_train[i]) in classNumber:
        path='dataset/train/'+str(int(y_train[i]))+'/'+str(i)+'.png'
        plt.imsave(path,x_train[i])

for i in range(10000):
    if int(y_test[i]) in classNumber:
        path='dataset/test/'+str(int(y_test[i]))+'/'+str(i)+'.png'
        plt.imsave(path,x_test[i])
