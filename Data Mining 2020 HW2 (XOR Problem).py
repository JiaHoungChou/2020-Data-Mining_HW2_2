import numpy as np
import pandas as pd
np.random.seed(123)

database= pd.DataFrame({"X_1": [0, 0, 1, 1], "X_2": [0, 1, 0, 1], "Label": [0, 1, 1, 0]})

def Step_Function_Matrix(X_matrix):
    X_matrix= np.array(X_matrix)
    for i in range(X_matrix.shape[0]):
        for j in range(X_matrix.shape[1]):
            if X_matrix[i][j]>= 0.50:
                X_matrix[i][j]= int(1)
            else:
                X_matrix[i][j]= int(0)
    return np.mat(X_matrix)

def sigmoid(X_matrix):
    X_matrix= np.array(X_matrix)
    for i in range(X_matrix.shape[0]):
        for j in range(X_matrix.shape[1]):
            X_matrix[i][j]= 1/ (1+ np.exp(-X_matrix[i][j]))
            
    return np.mat(X_matrix)

def cost_function(X_matrix):
    X_matrix= np.array(X_matrix)
    
    for i in range(X_matrix.shape[0]):
        for j in range(X_matrix.shape[1]):
            X_matrix[i][j]= np.array(X_matrix[i][j]* (1- X_matrix[i][j]))
    
    return np.array(X_matrix)

def accuracy_(prediction_, target_):
    correct= 0
    for i in range(0, len(prediction_)):
        if np.sum(np.absolute((np.array(prediction_[i])- np.array(target_[i]))))== 0:
            correct+= 1
        else:
            correct+= 0
    accuarcy= (correct/len(prediction_))* 100  
    return accuarcy

def forword_Backword_computation(X, label, eta, hidden_size, iteration):
    label= np.mat(label)
    X= np.hstack((np.ones((len(X), 1)), np.mat(X)))
    m, n= X.shape
    
    W_xh= np.mat(np.random.random(size= (hidden_size, n)))
    B_xh= np.mat(np.random.random(size=(m, hidden_size)))
    
    W_hy= np.mat(np.random.random(size= (label.shape[0], hidden_size)))
    B_hy= np.mat(np.random.random(size=(m, label.shape[0])))
    
    for i in range(1, iteration+ 1):
        net_h= X* W_xh.T- B_xh
        H_h= sigmoid(net_h)
    
        net_j= H_h* W_hy.T- B_hy
        y_j= sigmoid(net_j)

        deltas_j= np.mat(cost_function(X_matrix= y_j)* np.array(label.T- y_j))
        deltas_h= (((np.array(cost_function(X_matrix= H_h)))* np.array(np.sum(W_hy, axis= 0)))* np.array(np.sum(deltas_j, axis= 1)))

        W_hy+= eta* deltas_j.T* H_h
        B_hy+= -1* eta* deltas_j
        
        W_xh+= eta* deltas_h.T* X
        B_xh+= (-1* eta)* deltas_h
        
        class_matrix= Step_Function_Matrix(y_j)

        loss= accuracy_(prediction_= class_matrix, target_= label.T)
        
        if i % 100== 0:
            print("Iteration= %5d Training Accuracy= %4.2f"%(i, loss))
    
    return W_xh, B_xh, W_hy, B_hy

X= np.array(database[["X_1", "X_2"]])
y= np.array(database["Label"])

W_xh_, B_xh_, W_hy_, B_hy_= forword_Backword_computation(X= X, label= y, eta= 0.1, hidden_size= 4, iteration= 1000)

X_XOR= X
y_XOR= y

XOR_test_x_matrix= np.hstack((np.ones((len(X_XOR), 1)), np.mat(X_XOR)))
XOR_test_label= Step_Function_Matrix(sigmoid(sigmoid(XOR_test_x_matrix* W_xh_.T- B_xh_)* W_hy_.T- B_hy_))

print()
print("========== XOR Problem ===========")
print("*Test_label (Target= 0, 1, 1, 0):\n", np.array(XOR_test_label).ravel(), "\n")
print("BPN can solve the XOR problem, when test the model.")

