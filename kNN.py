__author__ = 'yz48'
import pandas as pd
import numpy as np
from collections import Counter


def getNeighors(train, test, k = 3):
    '''

    :param train: train data n*p nparray, last column is numeric: y
    :param test: test data m*p nparray, last column is numeric: y
    :param k: k parameter
    :return: prediction for test

    majority vote

    '''
    '''
    :param train:
    :param test:
    :param k:
    :return:
    '''
    res = []
    for i in range(len(test)):
        testInstance = test[i]
        distance = np.linalg.norm(train[:,:-1]-testInstance[:-1],axis = 1)
        index = distance.argsort()[:k]

        b = Counter(train[index,-1])
        res.append(b.most_common(1)[0][0])
    return res

def accuracy(x,y):
    if len(x) != len(y):
        return False
    else:
        cnt = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                cnt += 1
        return 1.0*cnt/len(x)


df = pd.read_csv('iris.csv', header = None)
ary = df.values

# data processing
print set(ary[:,-1])
d = {'Iris-virginica':1,'Iris-setosa':2, 'Iris-versicolor':3}
for row in ary:
    row[-1] = d[row[-1]]
df.replace(d, inplace = True)

# split data
np.random.seed(1)
msk = np.random.rand(len(df)) < 0.8
train = df[msk].values
test = df[~msk].values

print len(ary),len(train),len(test)

pred = getNeighors(train, test, k = 1)
print pred
print test[:,-1]
print accuracy(pred, test[:,-1])




