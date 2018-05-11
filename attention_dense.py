#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:36:49 2018

@author: zmj
"""
from keras.layers import Dense,multiply,Input,Dropout
from keras.models import Model
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
input_dim=32
drop_rate=0.5
def get_model():
  input_=Input(shape=(input_dim,))
  ##attention begins 
  attention_probs=Dense(input_dim,name="attention_probs",activation="softmax")(input_)
  x=multiply([input_,attention_probs])
  ##attention ends
  x=Dropout(drop_rate)(x)
  
  x=Dense(16)(x)
  
  x=Dense(1,activation="sigmoid")(x)
  
  model=Model(input_,x)
  
  model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
  return model

def get_data(n,attention_col):
  x=np.random.randn(n,input_dim)
  y=np.random.randint(0,2,size=n)
  for col in attention_col:
    x[:,col]=y/float(len(attention_col))
  return x,y

def get_activation(model,layer_name,inputs):
  layer=[l for l in model.layers if l.name==layer_name][0]
  
  func=K.function([model.input],[layer.output])
  
  return func([inputs])

if __name__=="__main__":
  x,y=get_data(100000,[1,10,20,29])
  
  m=get_model()
  
  m.fit(x,y,batch_size=100,epochs=20,validation_split=0.4)
  
  test_x=x[0,:].reshape(1,input_dim)
  attentions=get_activation(m,"attention_probs",test_x)[0].flatten()
  
  plt.bar(np.arange(input_dim),attentions)
  plt.title("attention vetors ")
  plt.show()
  
  
