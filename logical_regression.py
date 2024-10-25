'''
Author: xinao_seven_
Date: 2024-10-15 19:42:22
LastEditTime: 2024-10-24 20:27:04
LastEditors: xinao_seven_
Description: 
Encoding: utf-8
FilePath: /ML/logical_regression.py

'''

import numpy as np
import matplotlib.pyplot as plt
import copy,math
import pandas as pd


def compte_Fwx(w:np.ndarray,x_i:np.ndarray,b:float) ->float:
    return np.dot(w,x_i)+b


def sigmoid(z:float)->float:
    return 1/(1+np.exp(-z))


def compute_loss_reg(x:np.ndarray,y:np.ndarray,w:np.ndarray,b:float,lambda_:float=0) ->float:
    n,m = x.shape

    loss  =.0
    for i in range(n):
        
        f_wb_i = (sigmoid(compte_Fwx(w,x[i],b)))
        loss += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    reg_cost = 0
    for j in range(m):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*n)) * reg_cost 
    return loss/n

def compute_grident_reg(x:np.ndarray,y:np.ndarray,w:np.ndarray,b:float,lambda_:float=0):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = sigmoid(compte_Fwx(w,x[i],b))-y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j]+err*x[i,j]
        dj_db = err + dj_db

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_dw/m,dj_db/m

def predict(w:np.ndarray,x:np.ndarray,b:float)->int:
    probability = sigmoid(compte_Fwx(w,x,b))
    return 1 if probability>=0.5 else 0


def grident_decent(x:np.ndarray,y:np.ndarray,w_init:np.ndarray,b_init:float,iter:int,alpha:float,lambda_:float=0):
    
    loss_history = []
    w_final = copy.deepcopy(w_init)
    b_final = b_init

    for i in range(iter):
        dj_dw,dj_db = compute_grident_reg(x,y,w_final,b_final,0)

        w_final = w_final - alpha*dj_dw
        b_final = b_final - alpha*dj_db

        if i<100000:      
            loss_history.append( compute_loss_reg(x, y, w_final, b_final))

        if i% math.ceil(iter / 100) == 0:
            print(f"Iteration {i:4d}: Cost {loss_history[-1]} ")
        
    return w_final,b_final,loss_history

def main():
    data = pd.read_csv("./data/ex2data1.txt", header=None, names=['Size', 'Bedrooms', 'Price'])
    
    # data = (data - data.mean()) / data.std()
    cols = data.shape[1]
    X = data.iloc[:, :cols-1]
    Y = data.iloc[:, cols-1:cols]   
    X_train = np.array(X)
    y_train = np.array(Y)
    w_init = np.zeros_like(X_train[0])
    # w_init = np.array([0.03,0.029])
    b_init = 0#-6.76

    # class var
    np.random.seed(1)
    intial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -8

    iterations = 10000
    alpha = 0.001

    w,b,loss_his = grident_decent(X_train,y_train,intial_w,initial_b,iterations,alpha,0)

    True_cnt = 0
    print(f"b,w found by gradient descent: {b},{w} ")
    fig,(ax1,ax2) = plt.subplots(1,2,constrained_layout=True, figsize=(12, 4))
    # plot the probability 
    

    # Plot the original data
    ax1.set_ylabel(r'$x_1$')
    ax1.set_xlabel(r'$x_0$')   
    # ax.axis([0, 4, 0, 3.5])
    zero_data = []
    one_data = []
    for i in range(len(y_train)):
        if y_train[i] == predict(w,X_train[i],b):
            True_cnt += 1
        if y_train[i] == 0:
            zero_data.append(X_train[i])
        else:
            one_data.append(X_train[i])
    print(f"The Accuracy is {True_cnt/len(y_train)}%")
    zero = np.array(zero_data)
    one = np.array(one_data)
    ax1.scatter(zero[:,0],zero[:,1],color = "#0A4108")
    ax1.scatter(one[:,0],one[:,1],color = "#5383B5")
    # Plot the decision boundary
    x0 = -b/w[1]
    x1 = -b/w[0]
    ax1.plot([0,x0[0]],[x1[0],0], lw=1)

    ax2.plot(loss_his[1000:])
    plt.show()
    
                   
if __name__ == "__main__":
    main()