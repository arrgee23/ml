import network

from sklearn import datasets
import numpy as np

## Read data
def readAndSplitData():
    """ Read the data into suitable format
        split into train and test sets
    """
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


train,test = readAndSplitData()

#print(train)

nn = network.Network([4,5,3],is_from_file=False)
nn.SGD(train,500,10,0.5,test)