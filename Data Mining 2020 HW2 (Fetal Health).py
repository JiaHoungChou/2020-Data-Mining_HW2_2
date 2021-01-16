import numpy as np
import pandas as pd
import random
random.seed(123)
np.random.seed(123)

path= "/Users/ericchou/Desktop/PyCharm/Data Mining HW PROGRAM/fetal_health.csv"

with open(path, "r") as file:
    database= pd.read_csv(file)

def shuffle(x):
    list_= list(np.arange(0, len(x)).astype(int))
    random.shuffle(list_) 
    x= x.reindex(list_)
    return x

def one_hot_encoding(Dataframe_type, index_name):
    n= len(Dataframe_type[index_name].value_counts().index)
    category_index= np.array(Dataframe_type[index_name].value_counts().index)
    variables= []
    for i in range(0, len(category_index)):
        word= category_index[i]
        if len(word.split('"'))== 3:
            word= word.split('"')[1]
        variables.append(word)
    variables= np.array(variables)
    
    one_hot_encoding= []
    for i in range(0, len(Dataframe_type)):
        category_data= Dataframe_type[index_name][i]
        one_hot_encoding_row= []
        for j in range(0, n):
            if category_data == category_index[j]:
                one_hot_encoding_row.append(1)
            else:
                one_hot_encoding_row.append(0)                
        one_hot_encoding.append(one_hot_encoding_row)      
    one_hot_encoding= np.array(one_hot_encoding)   
    
    new_dataframe= pd.DataFrame(one_hot_encoding, columns= variables)
    return new_dataframe

def replace_one_hot_variable(dataframe_type, index_name, new_dataframe):
    index= dataframe_type.columns.tolist()   
    n= len(new_dataframe.columns)
    
    for i in range(0, n):
        index.insert(index.index(index_name)+ i, new_dataframe.columns[i])
        
    dataframe_type= dataframe_type.reindex(columns= index)
    dataframe_type= dataframe_type.drop([index_name], axis= 1)    
    for j in range(0, n):
        dataframe_type[new_dataframe.columns[j]]= new_dataframe[new_dataframe.columns[j]]
        
    return dataframe_type

database["fetal_health"]= database["fetal_health"].replace(1, "Normal").replace(2, "Suspect").replace(3, "Pathological")
fetal_health_dataframe= one_hot_encoding(database, "fetal_health")

database= shuffle(database).drop(["severe_decelerations"], axis= 1)
database= replace_one_hot_variable(dataframe_type= database, index_name= "fetal_health", new_dataframe= fetal_health_dataframe)

X= np.array(database.iloc[:, :-3])


# ### 1.Normal 2.Suspect 3.Pathological
y= np.array(database.iloc[:, -3:])

Training_Length= 2000
X_train= X[: Training_Length, :]
y_train= y[: Training_Length, :]

X_test= X[Training_Length:, :]
Target= y[Training_Length:, :]

def Classification_Function_Matrix(X_matrix):
    X_matrix= np.array(X_matrix)
    for i in range(X_matrix.shape[0]):
        for j in range(X_matrix.shape[1]):
            if X_matrix[i][j]== X_matrix[i][np.where(X_matrix[i]== np.max(X_matrix[i]))]:
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
        if np.sum(np.absolute(np.array(prediction_[i])- np.array(target_[i])))== 0:
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
    B_xh= np.mat(np.random.random(size=(1, hidden_size)))
    
    W_hy= np.mat(np.random.random(size= (label.shape[1], hidden_size)))
    B_hy= np.mat(np.random.random(size=(1, label.shape[1])))
    
    for i in range(1, iteration+ 1):
        net_h= X* W_xh.T- B_xh
        H_h= sigmoid(net_h)
    
        net_j= H_h* W_hy.T- B_hy
        y_j= sigmoid(net_j)
        
        deltas_j= np.array(cost_function(X_matrix= y_j)* np.array(label- y_j))
        deltas_h= (((np.array(cost_function(X_matrix= H_h)))* np.array(np.sum(W_hy, axis= 0)))* np.array(np.sum(deltas_j, axis= 1)[:, np.newaxis]))
        
        W_hy+= eta* deltas_j.T* H_h
        B_hy+= -1* eta* np.sum(deltas_j, axis= 0)
        
        W_xh+= eta* deltas_h.T* X
        B_xh+= (-1* eta)* np.sum(deltas_h, axis= 0)
        
        class_matrix= Classification_Function_Matrix(y_j)

        loss= accuracy_(prediction_= class_matrix, target_= label)

        if i % 100== 0:
            print("Iteration= %5d Training Accuracy= %4.2f"%(i, loss))
    
    return W_xh, B_xh, W_hy, B_hy

W_xh_, B_xh_, W_hy_, B_hy_= forword_Backword_computation(X= X_train, label= y_train, eta= 0.01, hidden_size= 10, iteration= 5000)

### Test the performance of model for training dataset
X_train= np.hstack((np.ones((len(X_train), 1)), np.mat(X_train)))
training_prediction= sigmoid(sigmoid(X_train* W_xh_.T- B_xh_)* W_hy_.T- B_hy_)
training_prediction= Classification_Function_Matrix(training_prediction)
accuarcy_train= accuracy_(prediction_= training_prediction, target_= y_train)
print("Training Accuarcy: %4.2f"%(accuarcy_train), "%")


### Test the performance of model
X_test= np.hstack((np.ones((len(X_test), 1)), np.mat(X_test)))
prediction= sigmoid(sigmoid(X_test* W_xh_.T)* W_hy_.T)
prediction= Classification_Function_Matrix(prediction)
accuarcy= accuracy_(prediction_= prediction, target_= Target)

print("Test Accuarcy: %4.2f"%(accuarcy), "%")