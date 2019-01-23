#-*- coding:utf-8 -*-
#author:zhangwei

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input , Conv2D , MaxPooling2D , Dense , AveragePooling2D
from keras.layers import BatchNormalization , Dropout , regularizers , Reshape , Flatten , Activation , concatenate
import os
from keras.optimizers import SGD , RMSprop , Adam

batch_size = 32
num_classes = 10
epoches = 100
data_augmentation = True
num_predictions = 100

(x_train , y_train) , (x_test , y_test) = mnist.load_data()
# print(x_train.dtype)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# print(x_train.dtype)

x_train /= 255
x_test /= 255
# print(x_train[0])
# print(y_train[0])
x_train = x_train.reshape([60000 , 28 , 28 , 1])
x_test = x_test.reshape([10000 , 28 , 28 , 1])
# print(x_test[0])

y_train = np_utils.to_categorical(y_train , num_classes=10)
y_test = np_utils.to_categorical(y_test , num_classes=10)
# print(y_train[0])

IMAGE_SIZE = 28
N_CHANNELS = 1
K = 12

def dense_block(input_tensor , channels):
    bn1 = BatchNormalization(epsilon=1e-4)(input_tensor)
    relu = Activation(activation='relu')(bn1)
    conv1 = Conv2D(filters=4*channels , kernel_size=[1 , 1] , padding='same')(relu)
    bn2 = BatchNormalization(epsilon=1e-4)(conv1)
    relu2 = Activation(activation='relu')(bn2)
    conv2 = Conv2D(filters=channels , kernel_size=[3 , 3] , padding='same')(relu2)
    return conv2

def transition_layer(input_tensor , channels):
    conv = Conv2D(filters=channels , kernel_size=[1 , 1] , padding='same')(input_tensor)
    pool = AveragePooling2D(pool_size=[2 , 2] , strides=[2 , 2])(conv)
    return pool

input_img = Input(shape=[IMAGE_SIZE , IMAGE_SIZE , N_CHANNELS] , name="Image_Input")
conv1 = Conv2D(filters=K*2 , kernel_size=[3 , 3] , padding='same')(input_img)

x = MaxPooling2D(strides=[2 , 2])(conv1)
b1_1 = dense_block(x , K)
b1_1_conc = concatenate([x , b1_1] , axis=-1)
b1_2 = dense_block(b1_1_conc , K)
b1_2_conc = concatenate([x , b1_1 , b1_2] , axis=-1)
b1_3 = dense_block(b1_2_conc , K)
b1_3_conc = concatenate([x , b1_1 , b1_2 , b1_3] , axis=-1)
b1_4 = dense_block(b1_3 , K)
b1_4_conc = concatenate([x , b1_1 , b1_2 , b1_3 , b1_4] , axis=-1)
b1_5 = dense_block(b1_4_conc , K)
b1_5_conc = concatenate([x , b1_1 , b1_2 , b1_3 , b1_4 , b1_5] , axis=-1)
b1_6 = dense_block(b1_5_conc , K)
x2 = transition_layer(b1_6 , K)

b2_1 = dense_block(x2 , K)
b2_1_conc = concatenate([x2 , b2_1] , axis=3)
b2_2 = dense_block(b2_1_conc , K)
b2_2_conc = concatenate([x2 , b2_1 , b2_2] , axis=-1)
b2_3 = dense_block(b2_2_conc , K)
b2_3_conc = concatenate([x2 , b2_1 , b2_2 , b2_3] , axis=3)
b2_4 = dense_block(b2_3_conc , K)
b2_4_conc = concatenate([x2 , b2_1 , b2_2 , b2_3 , b2_4] , axis=3)
b2_5 = dense_block(b2_4_conc , K)
b2_5_conc = concatenate([x2 , b2_1 , b2_2 , b2_3 , b2_4 , b2_5] , axis=3)
b2_6 = dense_block(b2_5_conc , K)
x3 = transition_layer(b2_6 , K)

b3_1 = dense_block(x3 , K)
b3_1_conc = concatenate([x3 , b3_1] , axis=3)
b3_2 = dense_block(b3_1_conc , K)
b3_2_conc = concatenate([x3 , b3_1 , b3_2] , axis=3)
b3_3 = dense_block(b3_2_conc , K)
b3_3_conc = concatenate([x3 , b3_1 , b3_2 , b3_3] , axis=3)
b3_4 = dense_block(b3_3_conc , K)
b3_4_conc = concatenate([x3 , b3_1 , b3_2 , b3_3 , b3_4] , axis=3)
b3_5 = dense_block(b3_4_conc , K)
b3_5_conc = concatenate([x3 , b3_1 , b3_2 , b3_3 , b3_4 , b3_5] , axis=3)
b3_6 = dense_block(b3_5_conc , K)
pool_f = AveragePooling2D(pool_size=[3 , 3] , strides=None , padding='valid')(b3_6)

flatten_p = Flatten()(pool_f)
logits = Dense(units=10 , use_bias=True , activation='softmax')(flatten_p)

model = Model(inputs=input_img , outputs=logits)
# model.summary()
adam = Adam(lr=0.01 , beta_1=0.9 , beta_2=0.99)
model.compile(optimizer=adam , loss='categorical_crossentropy' , metrics=['accuracy'])
model.fit(x=x_train , y=y_train , batch_size=batch_size , epochs=epoches , validation_split=0.2)

score = model.evaluate(x=x_test , y=y_test)
print("Test Loss" , score[0])
print("Test Accuracy" , score[1])



