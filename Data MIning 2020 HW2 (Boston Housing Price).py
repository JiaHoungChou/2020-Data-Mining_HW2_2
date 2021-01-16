import numpy as np
import pandas as pd
import random
random.seed(123)
np.random.seed(123)

path= "/Users/ericchou/Desktop/PyCharm/Data Mining HW PROGRAM/Boston_Housing.csv"
with open(path, "r") as file:
    database= pd.read_csv(file)

def shuffle(x):
    list_= list(np.arange(0, len(x)).astype(int))
    random.shuffle(list_) 
    x= x.reindex(list_)
    return x

def one_hot_encoding(Dataframe_type, index_name):
    n = len(Dataframe_type[index_name].value_counts().index)
    category_index = np.array(Dataframe_type[index_name].value_counts().index)
    variables = []
    for i in range(0, len(category_index)):
        word = category_index[i]
        variables.append(index_name+ "_"+ str(word))
    variables = np.array(variables)

    one_hot_encoding = []
    for i in range(0, len(Dataframe_type)):
        category_data = Dataframe_type[index_name][i]
        one_hot_encoding_row = []
        for j in range(0, n):
            if category_data == category_index[j]:
                one_hot_encoding_row.append(1)
            else:
                one_hot_encoding_row.append(0)
        one_hot_encoding.append(one_hot_encoding_row)
    one_hot_encoding = np.array(one_hot_encoding)

    new_dataframe = pd.DataFrame(one_hot_encoding, columns= variables)
    return new_dataframe

def replace_one_hot_variable(dataframe_type, index_name, new_dataframe):
    index = dataframe_type.columns.tolist()
    n = len(new_dataframe.columns)

    for i in range(0, n):
        index.insert(index.index(index_name) + i, new_dataframe.columns[i])

    dataframe_type = dataframe_type.reindex(columns=index)
    dataframe_type = dataframe_type.drop([index_name], axis=1)
    for j in range(0, n):
        dataframe_type[new_dataframe.columns[j]] = new_dataframe[new_dataframe.columns[j]]

    return dataframe_type

database= shuffle(database)

CHAS_dataframe= one_hot_encoding(Dataframe_type= database, index_name= "CHAS")
database= replace_one_hot_variable(dataframe_type= database, index_name= "CHAS", new_dataframe= CHAS_dataframe)

def normalization(X):
    X_std= np.array(pd.DataFrame(X.std(axis= 0)).replace(0.0, 1)).ravel()
    Z= (X- X.mean(axis= 0))/ X_std
    return Z, X.mean(axis= 0), X_std

X= np.array(database.iloc[:, : -1])
y= np.array(database.iloc[:, -1: ])

Training_Length= 354
X_train, _, _= normalization(X[: Training_Length, :])
y_train, y_train_mean, y_train_std= normalization(y[: Training_Length])


X_test, _, _= normalization(X[Training_Length:, :])
Target= y[Training_Length: ].ravel()

def sigmoid(X_matrix_sigmoid):
    X_matrix_sigmoid= np.array(X_matrix_sigmoid)
    for i in range(X_matrix_sigmoid.shape[0]):
        for j in range(X_matrix_sigmoid.shape[1]):
            X_matrix_sigmoid[i][j]= 1/ (1+ np.exp(-X_matrix_sigmoid[i][j]))
    return np.mat(X_matrix_sigmoid)

def cost_function(X_matrix):
    X_matrix= np.array(X_matrix)
    for i in range(X_matrix.shape[0]):
        for j in range(X_matrix.shape[1]):
            X_matrix[i][j]= np.array(X_matrix[i][j]* (1- X_matrix[i][j]))

    return np.array(X_matrix)


def MSE(X_matrix, y_matrix):
    X_matrix= np.array(X_matrix).ravel()
    y_matrix= np.array(y_matrix).ravel()
    return np.sum((y_matrix- X_matrix)**2)/ len(y_matrix)


def forword_Backword_computation(X, target, eta, hidden_size, iteration):
    target= np.mat(target).ravel()
    X= np.hstack((np.ones((len(X), 1)), np.mat(X)))
    m, n= X.shape
    
    W_xh= np.mat(np.random.random(size= (hidden_size, n)))
    B_xh= np.mat(np.random.random(size=(1, hidden_size)))
    
    W_hy= np.mat(np.random.random(size= (target.shape[0], hidden_size)))
    B_hy= np.mat(np.random.random(size= (1, target.shape[0])))
    
    for i in range(1, iteration+ 1):
        net_h= X* W_xh.T- B_xh
        H_h= sigmoid(net_h)
        
        net_j= H_h* W_hy.T- B_hy
        
        deltas_j= (1/ H_h.shape[0])* (np.array(target.T)- np.array(net_j))
        deltas_h= (1/ net_j.shape[0])* (np.array(target.T)- np.array(H_h))* (np.array(H_h)* (1- np.array(H_h)))
        
        W_hy+= eta* deltas_j.T* H_h
        B_hy+= -eta* np.sum(deltas_j, axis= 0)
        
        W_xh+= eta* deltas_h.T* X
        B_xh+= -eta* np.sum(deltas_h, axis= 0)
        
        if i % 500== 0:
            print("Iteration= %5d, Mean Square Error %4.2f"%(i, MSE(net_j, target)))
            
    return W_xh, np.array(B_xh), W_hy, np.array(B_hy)

W_xh, B_xh, W_hy, B_hy= forword_Backword_computation(X= X_train, target= y_train, eta= 0.01, hidden_size= 10, iteration= 10000)


### Training the performance of model for training dataset
X_train= np.hstack((np.ones((len(X_train), 1)), np.mat(X_train)))
y_hat_train= np.array(sigmoid(X_train* W_xh.T- B_xh)* W_hy.T- B_hy).ravel()
y_hat_train= (y_hat_train* y_train_std)+ y_train_mean
y_true_train= y[: Training_Length].ravel()
print("Mean Square Error : ", round(MSE(y_hat_train, y_true_train), 4), " ( Training Set )")

### Test the performance of model for test dataset
X_test= np.hstack((np.ones((len(X_test), 1)), np.mat(X_test)))
y_hat= np.array(sigmoid(X_test* W_xh.T- B_xh)* W_hy.T- B_hy).ravel()
y_hat= (y_hat* y_train_std)+ y_train_mean
y_true= Target.ravel()
print("Mean Square Error : ", round(MSE(y_hat, y_true), 4), " ( Test Set )")