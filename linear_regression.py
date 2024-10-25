'''
Author: xinao_seven_
Date: 2024-10-10 22:49:04
LastEditTime: 2024-10-25 15:43:09
LastEditors: xinao_seven_
Description: 
Encoding: utf-8
FilePath: /ML/linear_regression.py

'''

import numpy as np
import matplotlib.pyplot as plt
import copy,math
import pandas as pd



def compute_Fwx(w:np.ndarray,x:np.ndarray,b:float) :
    return np.dot(w,x)+b

def compute_loss(x:np.ndarray,y:np.ndarray,w:np.ndarray,b:float):
    n = x.shape[0]

    loss  =.0
    for i in range(n):
        loss = loss + (compute_Fwx(w,x[i],b)-y[i])**2
        loss = loss /(2*n)
    return  loss

def compute_loss_reg(x:np.ndarray,y:np.ndarray,w:np.ndarray,b:float,lambda_:float):
    n,m = x.shape
    loss  =.0
    for i in range(n):
        loss = loss + (compute_Fwx(w,x[i],b)-y[i])**2
        loss = loss /(2*n)
        reg_cost = 0
    for j in range(m):
        reg_cost += (w[j]**2)                                         
    reg_cost = (lambda_/(2*n)) * reg_cost 
    return  loss+reg_cost


def compute_grident_reg(x:np.ndarray,y:np.ndarray,w:np.ndarray,b:float,lambda_:float =1):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = compute_Fwx(w,x[i],b)-y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j]+err*x[i,j]
        dj_db = err + dj_db
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
    return dj_dw/m,dj_db/m

def grident_decent(x:np.ndarray,y:np.ndarray,w_init:np.ndarray,b_init:float,iter:int,alpha:float):
    
    loss_history = []
    w_final = copy.deepcopy(w_init)
    b_final = b_init

    for i in range(iter):
        dj_dw,dj_db = compute_grident_reg(x,y,w_final,b_final)

        w_final = w_final - alpha*dj_dw
        b_final = b_final - alpha*dj_db

        if i<100000:      
            loss_history.append( compute_loss_reg(x, y, w_final, b_final,1))

        if i% math.ceil(iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {loss_history[-1]} ")
        
    return w_final,b_final,loss_history

def main():
    data = pd.read_csv("./data/ex1data2.txt", header=None, names=['Size', 'Bedrooms', 'Price'])
    
    # data = (data - data.mean()) / data.std()
    cols = data.shape[1]
    X = data.iloc[:, :cols-1]
    Y = data.iloc[:, cols-1:cols]   
    X_train = np.array(X)
    y_train = np.array(Y)

    w_init = np.zeros_like(X_train[0])
    b_init = 0.
    iterations = 1000
    alpha = 5.0e-9

    w,b,loss_his = grident_decent(X_train,y_train,w_init,b_init,iterations,alpha)

    print(f"b,w found by gradient descent: {b},{w} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w) + b}, target value: {y_train[i]}")
    fig, ax1= plt.subplots(1, 1, constrained_layout=True, figsize=(6, 4))
    ax1.plot(loss_his)
    ax1.set_title("Cost vs. iteration") 
    ax1.set_ylabel('Cost')              
    ax1.set_xlabel('iteration step')     
    plt.show()
                   
if __name__ == "__main__":
    main()