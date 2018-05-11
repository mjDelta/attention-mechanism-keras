#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:56:03 2018

@author: zmj
"""
import numpy as np
from keras.layers import Dense,LSTM,Input,multiply,Permute,RepeatVector,Flatten
from keras.models import Model
import keras.backend as K
from matplotlib import pyplot as plt

TIME_STEPS=30
N=2

def get_data(n,target_time=10):
  x=np.random.randn(n,TIME_STEPS,N)
  y=np.random.randint(0,2,size=(n,1))
  x[:,target_time,:]=np.tile(y,(1,N))
  return x,y

def attention_block(inputs):
  x=Permute((2,1))(inputs)
  x=Dense(TIME_STEPS,activation="softmax")(x)
  x=Permute((2,1),name="attention_prob")(x)
  x=multiply([inputs,x])
  return x

def get_model_before_lstm():
  input_=Input(shape=(TIME_STEPS,N))
  x=attention_block(input_)
  x=LSTM(32)(x)
  x=Dense(1)(x)
  model=Model(input_,x)
  model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
  return model

def get_activation(model,layer_name,inputs):
  layer=[l for l in model.layers if l.name==layer_name][0]
  
  func=K.function([model.input],[layer.output])
  
  return func([inputs])[0]

if __name__=="__main__":
  model=get_model_before_lstm()
  
  x,y=get_data(10000)
  
  model.fit(x,y,epochs=20,validation_split=0.2)
  
  test_x,test_y=get_data(1)
  
  attention_probs=np.mean(get_activation(model,"attention_prob",test_x),axis=2).flatten()
  
  plt.bar(np.arange(TIME_STEPS),attention_probs)
  plt.title("LSTM attention probs")
  plt.show()
  
  

