# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:46:01 2019

@author: ktm
"""
#%% 사용할 모델 라이브러리 import
import sys, os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

#%%
def load_dataset():
    x_train, x_test, y_train, y_test = np.load(root_dir + "koreanfood02.npy")
    x_train = x_train.astype("float") / 256
    x_test = x_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, x_test, y_train, y_test

#%% CNN
# 모델 구성 (2)
def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='Same',
    input_shape=in_shape))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25)) # dropout
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('sigmoid'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # dropout
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('sigmoid'))
    # dropout
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])
    return model
#%% 모델 학습
# 모델 학습을 수행하고 저장된 모델을 파일로 저장 (3)
def model_train(x, y):
model = build_model(x.shape[1:])
model.fit(x, y, batch_size=32, epochs=20)
return model

#%% 모델 평가
# 모델 학습을 수행하고 저장된 모델을 파일로 저장 (3)
def model_train(x, y):
    model = build_model(x.shape[1:])
    model.fit(x, y, batch_size=32, epochs=20)
    return model

#%% 실행
# 모델 학습 및 평가
x_train, x_test, y_train, y_test = load_dataset()
model = model_train(x_train, y_train)
model_eval(model, x_test, y_test)
