__author__ = 'yz48'
import numpy as np

k = 3
def init_board_gauss(N, k=3):
    np.random.seed(1)
    n = float(N)/k
    X = []
    clist = [[0.7,0.3],[-0.4,0.5],[0.3,-0.8]]
    slist = [0.35,0.45,0.2]
    for i in range(k):
        c = clist[i]
        s = slist[i]
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

def cluster_points(X, mu):
    return

X = init_board_gauss(2000)

np.random.seed(1)
index = np.random.randint(0,len(X), size = k)
print index
oldmu = X[index]
iter = 10

for it in range(iter):
    classes = []
    for x in X:
        #print x
        c = np.linalg.norm(x-oldmu,axis = 1).argmin() # sqrt(sum(x^2+y^2+z^2+...))
        #print c, type(c)
        classes.append(c)
    classes = np.array(classes)

    mu = []
    for i in range(k):
        x = X[np.where(classes == i)]
        mu.append(np.sum(x,axis = 0)/len(x))
    if it %1 == 0:
        print it, mu
    oldmu = mu





