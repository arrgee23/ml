import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os

    
def trainTestSplit(df,testCol,testSize=0.2):
    """having passed a dataframe and a output col splits into 
        train and test set
    """
    y=data[testCol]
    x=data.drop(testCol,axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testSize)
    return x_train,x_test,y_train,y_test

def readFileMakeDF(fileName):
    data=pd.read_csv(fileName);
    df = pd.DataFrame(data);
    return df

importall()
df = readFileMakeDF("../input/train.csv")
x_train,x_test,y_train,y_test = trainTestSplit(df,'Survived')
x_train
