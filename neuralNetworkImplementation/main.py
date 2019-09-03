import network

from sklearn import datasets
import numpy as np
import pandas as pd

def read_and_split_data2():
    
    data = pd.read_csv("mnist.csv")
    
    

    x = data.values[:,1:]
    y = []
    
    
    labels = data.values[:,0]
    for out in labels:
        if(out==0):
            y.append([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
        elif(out==1):
            y.append([[0],[1],[0],[0],[0],[0],[0],[0],[0],[0]])
        elif(out==2):
            y.append([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]])
        elif(out==3):
            y.append([[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]])
        elif(out==4):
            y.append([[0],[0],[0],[0],[1],[0],[0],[0],[0],[0]])
        elif(out==5):
            y.append([[0],[0],[0],[0],[0],[1],[0],[0],[0],[0]])
        elif(out==6):
            y.append([[0],[0],[0],[0],[0],[0],[1],[0],[0],[0]])
        elif(out==7):
            y.append([[0],[0],[0],[0],[0],[0],[0],[1],[0],[0]])
        elif(out==8):
            y.append([[0],[0],[0],[0],[0],[0],[0],[0],[1],[0]])
        else:
            y.append([[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]])
        
          

    y = np.array(y)

    train_data_proportion = 0.7
    train = []
    test = []
    train_for_test = []
    examples = x.shape[0]
    train_size = examples*train_data_proportion
    count = 0

    arr = np.random.permutation(examples)

    #print(arr)

    
    for i in arr:
        e = x[i,:]
        #print(e)
        e = np.reshape(e,(784,1))
        #print(e.shape)
        if(count<train_size):
            train.append((e,y[i])) # train set expects y as a vector of last layer activations
            train_for_test.append((e,np.argmax(y[i])))
        else:
            
            test.append((e,np.argmax(y[i])))

        count+=1

    
    return train,test,train_for_test


## Read data
def readAndSplitData():
    """ Read the data into suitable format
        split into train and test sets
    """
    data = pd.read_csv("mnis")
    
    iris = datasets.load_iris()

    x = iris.data
    y = []
    for out in iris.target:
        if(out==0):
            y.append([[1],[0],[0]])
        elif(out==1):
            y.append([[0],[1],[0]])
        else:
            y.append([[0],[0],[1]])     

    y = np.array(y)

    train_data_proportion = 0.3
    train = []
    test = []

    examples = x.shape[0]
    train_size = examples*train_data_proportion
    count = 0

    arr = np.random.permutation(examples)

    #print(arr)

    
    for i in arr:
        e = x[i,:]
        #print(e)
        e = np.reshape(e,(4,1))
        #print(e.shape)
        if(count<train_size):
            train.append((e,y[i])) # train set expects y as a vector of last layer activations
        else:
            
            test.append((e,np.argmax(y[i])))

        count+=1


    return train,test


train,test,train_for_test = read_and_split_data2()

#print(train)

nn = network.Network([784,50,10],is_from_file=True)
nn.SGD(train,2,10,0.05,None)

print("Test matrix: ")
s1 = nn.evaluate2(test)
print("Test accuracy: ",s1/len(test))



print("Train matrix: ")
s2 = nn.evaluate2(train_for_test)
print("Train accuracy: ",s2/len(train_for_test))