import keras
from keras import optimizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation
from keras import layers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
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


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = ImageDataGenerator().flow_from_directory(
    'dataset/train/',
    target_size=(32,32),
    batch_size=20,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator =ImageDataGenerator().flow_from_directory(
    'dataset/test/',
    target_size=(32,32),
    batch_size=20,
    class_mode='categorical'
)

train_generator[1][0].shape # (20,32,32,3)


#model

model=Sequential()

model.add(layers.Conv2D(32,
                        (3,3),
                        activation='relu',
                        padding='same',
                        input_shape= (32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32,
                        (3,3),
                        padding='same',
                        activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64,
                        (3,3),
                        padding='same',
                        activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64,
                        (3,3),
                        padding='same',
                        activation='relu'))
model.add(layers.Conv2D(128,
                        (3,3),
                        padding='same',
                        activation='relu'))
model.add(layers.Conv2D(256,
                        (3,3),
                        padding='same',
                        activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(6,activation='softmax'))
model.summary()

#o modeli derle

model.compile(loss='mean_squared_error',
            optimizer=Adam(lr=0.001),
            metrics=['acc'])

history=model.fit(x_train,
                y_train,
                epochs=30,
                validation_data=(x_test,y_test))

score = model.evaluate(
    x=x_test,
    y=y_train
)

print("Test-accuracy: " + str(round(score[1], 3) * 100) + "%\n" + "Test-loss: " + str((score[0])))

model.save('cifar10_model1.h5')
model.save_weights('cifar10_weights1.h5')
model.load_weights('cifar10_weights1.h5')
from keras import models
models.load_model('cifar10_model1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

np.save('history1.npy', (acc, val_acc, loss, val_loss))
(acc,val_acc,loss,val_loss)=np.load('history.npy')

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss Value')
plt.xlabel('No. epoch')
plt.show()

plt.plot(history.history['val_acc'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()