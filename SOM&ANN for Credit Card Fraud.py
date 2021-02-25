# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:15:11 2021

@author: rohan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Credit_Card_Applications.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler(feature_range=(0,1))
X=mn.fit_transform(X)

from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X,num_iteration=100)

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar() 
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor=colors[y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
show()

fra=som.win_map(X)
fraud=np.concatenate((fra[(4,8)],fra[(2,7)]),axis=0)
frauds=mn.inverse_transform(fraud)

customers=df.iloc[:,1:].values
is_frauds=np.zeros(690)

for j in range(len(df)):
    if df.iloc[j,0] in frauds:
        is_frauds[j]=1
        
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(customers)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
ann=tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=2,activation='relu',input_dim=15))

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

ann.fit(X_train,is_frauds,batch_size=1,epochs=2)

pred=ann.predict(X_train)

y_pred=np.concatenate((df.iloc[:,0:1].values,pred),axis=1)

y_pre=y_pred[y_pred[:,1].argsort()]