'''
Author: xinao_seven_
Date: 2024-10-25 11:41:46
LastEditTime: 2024-10-25 12:08:24
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
# plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
# plt.show()
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
    X_train,y_train,
    epochs=10
)
p_nonpreferred = model.predict(X_train)
true_cnt = 0
for i in range(len(p_nonpreferred)):
    if y_train[i] == np.argmax(p_nonpreferred[i]):
        true_cnt += 1
print(f"The Accuracy is {true_cnt/len(y_train)}%")