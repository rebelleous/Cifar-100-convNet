# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 22:54:45 2021

@author: anilu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:09:51 2021
Cifar 10 veri setinin konvolüsyon ağı ile eğitimi
@author: anilu
"""
import keras
import os
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import layers
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


#Eğitim ve test veri setleri
base_dir = 'C:\\Users\\anilu\\Desktop\\Cifar100'
train_dir= os.path.join(base_dir,'train')
test_dir= os.path.join(base_dir,'test')

train_datagen = ImageDataGenerator(rescale=1./255) #Veriler 0-1 arasına ölceklendi
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(32,32), #Tüm görüntüler 32*32 boyutunda
    batch_size=20,  #Batch başına görüntü 20
    class_mode='categorical')  #Multiclass classification olduğu için


test_datagen = ImageDataGenerator(rescale=1./255) #Veriler 0-1 arasına ölceklendi
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32,32), #Tüm görüntüler 32*32 boyutunda
    batch_size=20, #Batch başına görüntü 20
    class_mode='categorical')  #Multiclass classification olduğu için

#Dropout olmayan model
def model1():
    model=Sequential()
    model.add(layers.Conv2D(32,
                        (3,3),
                        activation='relu',
                        padding='same',
                        input_shape = (32,32,3)))
    model.add(layers.Conv2D(64,
                        (3,3),
                        padding='same',
                        activation='relu'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(128,
                        (3,3),
                        padding='same',
                        activation='relu'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(128,
                        (3,3),
                        padding='same',
                        activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(128,activation='relu'))     
    model.add(layers.Dense(6,activation='softmax'))  
    #multiclass classification olduğu için çıkışımız softmaxtir.   
    model.summary()
    #  modeli derle
    from keras import optimizers
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    return model

#Dropout olan model
def model2():
    model=Sequential()
    model.add(layers.Conv2D(32,
                        (3,3),
                        activation='relu',
                        padding='same',
                        input_shape = (32,32,3)))
    model.add(layers.Conv2D(64,
                        (3,3),
                        padding='same',
                        activation='relu'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(128,
                        (3,3),
                        padding='same',
                        activation='relu'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Dropout(0,2)) 
    model.add(layers.Conv2D(128,
                        (3,3),
                        padding='same',
                        activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(128,activation='relu'))   
    model.add(layers.Dropout(0,2)) 
    #128*6 tane bağlantının %20 sini random olarak koparır.
    #Ezberlemeyi azaltmak için dropout eklendi
    model.add(layers.Dense(6,activation='softmax'))  
    #multiclass classification olduğu için çıkışımız softmaxtir.   
    model.summary()
    #modeli derle
    from keras import optimizers
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    return model


    model= model1()  # Dropout olmayan model atandı.

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 150, #bir epochta kaç güncelleme yapılacak 
        # total training Samples (500*6=3000) / TrainingbatchSize(20) = 150
        epochs=30, 
        validation_data=test_generator,
        validation_steps=30)
        # total validation samples (100*6=600) / ValidationBatchSize(20) = 30


    acc=history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1,len(acc)+1)
    
    plt.plot(epochs, acc, 'bo', label='Training Acc')
    plt.plot(epochs, val_acc, 'r', label='Validation Acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    model= model2()  # Dropout olan model atandı.

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 150, #bir epochta kaç güncelleme yapılacak 
        # total training Samples (500*6=3000) / TrainingbatchSize(20) = 150
        epochs=30, 
        validation_data=test_generator,
        validation_steps=30)
        # total validation samples (100*6=600) / ValidationBatchSize(20) = 30
        
        
    acc=history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1,len(acc)+1)
    
    plt.plot(epochs, acc, 'bo', label='Training Acc')
    plt.plot(epochs, val_acc, 'r', label='Validation Acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Data Augmentation / Veri Zenginleştirme



#Eğitim ve test veri setleri
base_dir = 'C:\\Users\\anilu\\Desktop\\Cifar100'
train_dir= os.path.join(base_dir,'train')
test_dir= os.path.join(base_dir,'test')

train_datagen = ImageDataGenerator(   
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(32,32), #Tüm görüntüler 32*32 boyutunda
    batch_size=20,  #Batch başına görüntü 20
    class_mode='categorical')  #Multiclass classification olduğu için

test_datagen = ImageDataGenerator() 
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32,32), #Tüm görüntüler 32*32 boyutunda
    batch_size=20, #Batch başına görüntü 20
    class_mode='categorical')  #Multiclass classification olduğu için


model= model1()  # Dropout olmayan model atandı.

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 150, #bir epochta kaç güncelleme yapılacak 
        # total training Samples (500*6=3000) / TrainingbatchSize(20) = 150
        epochs=30, 
        validation_data=test_generator,
        validation_steps=30)
        # total validation samples (100*6=600) / ValidationBatchSize(20) = 30

    acc=history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1,len(acc)+1)
    
    plt.plot(epochs, acc, 'bo', label='Training Acc')
    plt.plot(epochs, val_acc, 'r', label='Validation Acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


from keras_preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
# 7=beetle 24=cockroach 34=fox 59=pine_tree 82=sunflower 85=tank

tank= image.load_img('C:\\Users\\anilu\\Desktop\\Cifar100\\test\\85\\2639.png',
                        target_size=(32,32))
plt.imshow(tank)
# Numpy Dizisine Dönüştür
Giris=image.img_to_array(tank)
# Görüntüyü Ağa Uygula
y=model.predict(Giris.reshape(1,32,32,3))

#En Yüksek tahmin sınıfını bul
tahmin_index = np.argmax(y)
tahmin_yuzde= y[0][tahmin_index]*100
print(tahmin_yuzde)



# Grafik Yorumu Training Acc/ValidationAcc 
# Eğitimde kullanılan verileri ço iyi öğrenmi ama eğitimde kullanılmayan verielre
#  karşı başarım düşmüş. Yani overfitting olmuş.





