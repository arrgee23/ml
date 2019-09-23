#!/usr/bin/env python
# coding: utf-8

input("Enter no of features",D)
input("Enter no of features to select",d)

def fitness(feature):

    return sum(feature)

class tree:
    def __init__(self,features,DD,dd):
        self.root = None
        self.d = dd
        self.D = DD
        self.totalLevels = DD-dd+1
        self.features = features
        self.max = 0
        self.represent = {}
        for i in range(self.totalLevels):
            self.represent[i] = []
    
    def branch_and_bound(self):
        preserve = []
        self.root = grow_tree(self,self.features,preserve,0)

        
class treenode:
    def __init__(self,selected_features,curr_level):
        self.selected_features = selected_features
        self.curr_level = curr_level
        self.children = []



def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]


import random
def grow_tree(t,features,preserve,l):
    
    #print(features,preserve,l)
    n = treenode(features,l)
     #try to print t
    for i in range(l):
        print("\t",end='')
    print(n.selected_features)

    val = fitness(features)
    if(l>=t.totalLevels-1 or val<t.max): # bound and base case
        if(l==t.totalLevels-1): #root node
            if(val>t.max):
                t.max = val
                #print("***",val)
        return n
    else:
        
        no_children = t.d +1 -len(preserve)
        copy_preserve = preserve.copy()
        child_leave_feature = []
        child_preserve = []

        r = random.sample(diff(features,preserve),no_children)

        #buid children properties
        for i in range(no_children):
            #r = None
            #while(r in child_leave_feature):
            
            
            child_leave_feature.append(r[i])
            child_preserve.append(copy_preserve)
            copy_preserve = copy_preserve+[r[i]]

        for i in range(no_children-1,-1,-1):
            child_features = diff(features,[child_leave_feature[i]])
            #print(features,"------",[child_leave_feature[i]],"------",child_features)
            n.children.append(grow_tree(t,child_features,child_preserve[i],l+1))

        
        return n


#breakpoint()
t = tree([1,2,3,4,5,6],6,2)
t.branch_and_bound()
print(t.max)

