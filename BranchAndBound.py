#!/usr/bin/env python
# coding: utf-8
import numpy as np
from graphviz import Digraph

from graphviz import Graph
dot = Digraph(comment='branch and bound')

D = input("Enter no of features ")
d = input("Enter no of features to select ")    
k = input("Enter the features separated by space ")

D = int(D)
d = int(d)
k = k.split(" ")
fl = []
for numstr in k:
    fl.append(float(numstr))
#print(D,d,fl)

def fitness(feature):
    f = np.dot(feature,feature)
    #print(f)
    """sum = 0
    for i in range(f.shape[0]):
        sum += f[i]"""

    #print(sum)
    #return pow(3,f)
    return f

class tree:
    def __init__(self,features,DD,dd):
        self.root = None
        self.d = dd
        self.D = DD
        self.totalLevels = DD-dd+1
        self.features = features
        self.max = 0
        self.best_features = None
        self.represent = {}
        for i in range(self.totalLevels):
            self.represent[i] = []
    
    def branch_and_bound(self):
        preserve = []
        self.root = grow_tree(self,self.features,preserve,0,None)

        
class treenode:
    def __init__(self,selected_features,curr_level):
        self.selected_features = selected_features
        self.curr_level = curr_level
        self.children = []



def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def minimum(list1,list2):
    if(list1>list2):
        return list2
    else:
        return list1

import random
def grow_tree(t,features,preserve,l,leave_feature):
    
    val = fitness(features)
    if(val<=t.max):
        return None
    #print(features,preserve,l)
    n = treenode(features,l)
     #try to print t
    #print(leave_feature,end=' ')
    for i in range(l):
        print("---- ",end='')
    print(leave_feature,'----',n.selected_features)

    dot.node(str(n.selected_features),str(n.selected_features))
    
    if(l>=t.totalLevels-1 or val<=t.max): # bound and base case
        if(l==t.totalLevels-1): #root node
            if(val>t.max):
                t.max = val
                t.best_features = features
                #print("***",val)
        return n
    else:
        
        no_children = t.d +1 -len(preserve)
        copy_preserve = preserve.copy()
        child_leave_feature = []
        child_preserve = []
        
        m = minimum(no_children,len(diff(features,preserve)))
        r = random.sample(diff(features,preserve),m)
        r.sort(reverse=True)
        #buid children properties
        for i in range(m):
            #r = None
            #while(r in child_leave_feature):
            
            
            child_leave_feature.append(r[i])
            child_preserve.append(copy_preserve)
            copy_preserve = copy_preserve+[r[i]]

        for i in range(m-1,-1,-1):
            child_features = diff(features,[child_leave_feature[i]])
            #print(features,"------",[child_leave_feature[i]],"------",child_features)
            if(fitness(n.selected_features)>t.max):
                dot.edge(str(n.selected_features),str(child_features),label=str(child_leave_feature[i]))
            n.children.append(grow_tree(t,child_features,child_preserve[i],l+1,r[i]))

        
        return n


#breakpoint()
t = tree(fl,D,d)
print("\n-------The Branch and Bound tree------")
t.branch_and_bound()
print("Best features are: ",t.best_features)

dot.render('save.png',format='png')