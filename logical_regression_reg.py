'''
Author: xinao_seven_
Date: 2024-10-16 16:16:41
LastEditTime: 2024-10-24 20:26:50
LastEditors: xinao_seven_
Description: 
Encoding: utf-8
FilePath: /ML/logical_regression_reg.py

'''

import numpy as np
import matplotlib.pyplot as plt
import copy,math
import pandas as pd



def compte_Fwx(w:np.ndarray,x_i:np.ndarray,b:float) ->float:
    return np.dot(w,x_i)+b


def sigmoid(z:float)->float:
    return 1/(1+np.exp(-z))

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

def compute_loss_reg(x:np.ndarray,y:np.ndarray,w:np.ndarray,b:float,lambda_:float=0) ->float:
    n,m = x.shape

    loss  =.0
    for i in range(n):
        
        f_wb_i = (sigmoid(compte_Fwx(w,x[i],b)))
        loss += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    reg_cost = 0
    for j in range(m):
        reg_cost += (w[j]**2)                                          
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

        if i% math.ceil(iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {loss_history[-1]} ")
        
    return w_final,b_final,loss_history

def plot_data(X_train, y_train, pos_label="y=1", neg_label="y=0"):
    zero_data = []
    one_data = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            zero_data.append(X_train[i])
        else:
            one_data.append(X_train[i])
    
    zero = np.array(zero_data)
    one = np.array(one_data)
    plt.scatter(zero[:,0],zero[:,1])
    plt.scatter(one[:,0],one[:,1])

def plot_decision_boundary(w, b, X, y):
    # Credit to dibgerge on Github for this plotting code
     
    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)
        
        # important to transpose z before calling contour       
        z = z.T
        
        # Plot z = 0
        plt.contour(u,v,z, levels = [0.5], colors="g")

def main():
    data = pd.read_csv("./data/ex2data2.txt")
    
    cols = data.shape[1]
    X = data.iloc[:, :cols-1]
    Y = data.iloc[:, cols-1:cols]
    
    X_train = np.array(X)
    y_train = np.array(Y)
    mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])

    # class var
    np.random.seed(1) 
    initial_w  = np.random.rand(mapped_X.shape[1]) - 0.5 
    initial_b = 0.5

    iterations = 10000
    alpha = 0.01
    lambda_ = 0.01
    w,b,loss_his = grident_decent(mapped_X,y_train,initial_w,initial_b,iterations,alpha,lambda_)

    True_cnt = 0
    print(f"b,w found by gradient descent: {b},{w} ")

    # ax.axis([0, 4, 0, 3.5])

    for i in range(len(y_train)):
        if y_train[i] == predict(w,mapped_X[i],b):
            True_cnt += 1
    
    print(f"The Accuracy is {True_cnt/len(y_train)}%")
    # zero = np.array(zero_data)
    # one = np.array(one_data)
    # ax1.scatter(zero[:,0],zero[:,1],color = "#0A4108")
    # ax1.scatter(one[:,0],one[:,1],color = "#5383B5")
    # Plot the decision boundary
    # x0 = -b/w[1]
    # x1 = -b/w[0]
    # ax1.plot([0,x0[0]],[x1[0],0], lw=1)

    # ax2.plot(loss_his[1000:])
    # plt.show()
    plot_decision_boundary(w, b, mapped_X, y_train)
    plt.show()
    
                   
if __name__ == "__main__":
    main()