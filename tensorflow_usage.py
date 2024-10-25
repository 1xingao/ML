'''
Author: xinao_seven_
Date: 2024-10-25 11:41:46
LastEditTime: 2024-10-25 17:54:28
LastEditors: xinao_seven_
Description: 
Encoding: utf-8
FilePath: /ML/tensorflow_usage.py

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30) # type: ignore

model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train[:1700],y_train[:1700],
    epochs=10
)

[layer1, layer2, layer3] = model.layers
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()

# input x:y is [(2000*2):]->[(x,y):class]
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
# a[2] is (2000*25)
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
# a[3] is (2000*15)
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
print(X_train.shape)

p_nonpreferred = model.predict(X_train[1700:])
true_cnt = 0
for i in range(len(p_nonpreferred)):
    if y_train[i+1700] == np.argmax(p_nonpreferred[i]):
        true_cnt += 1
print(f"The Accuracy is {true_cnt/len(y_train[1700:])}%")

plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.show()