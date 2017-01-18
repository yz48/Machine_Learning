__author__ = 'yz48'
import numpy as np
#from pylab import scatter, show, title, xlabel, ylabel, plot, contour
class machine_learning:

    def linear_regression_one_dimension(self,x,y):
        n = len(x)
        if n != len(y):
            return 'x and y dimension miss match'

        cov = sum((x-np.mean(x))*(y-np.mean(y)))
        sx = sum((x-np.mean(x))**2)
        b = cov/(sx)
        a = np.mean(y)-b*np.mean(x)
        return a,b

    def linear_regression_grad_des(self,x,y,alpha,iterations):
        n = len(y)
        X = np.vstack([x,np.ones(n)])
        theta = np.zeros((1,2))
        for i in range(iterations):
            pred = np.dot(theta,X).flatten()
            errors_x1 = (pred - y) * X[0,:]
            errors_x2 = (pred - y) * X[1,:]
            theta[0][0] = theta[0][0] - alpha* errors_x1.sum()
            theta[0][1] = theta[0][1] - alpha* errors_x2.sum()
            if iterations%500000 == 0:
                print sum((np.dot(theta,X).flatten() - y)**2)
        return theta

    def linear_regression_matrix(self,x,y):
        X = np.vstack([np.ones(n),x]).T
        y = np.array(y)[np.newaxis].T
        theta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
        return theta

m = machine_learning()

# linear regression
n = 100
x = range(n)
a = 25
b = 6
mu = 0
sigma = 3
y = a+b*np.array(x)+np.random.normal(mu, sigma, n)
print m.linear_regression_one_dimension(x,y)
print m.linear_regression_grad_des(x,y,0.000001,200000)
print m.linear_regression_matrix(x,y)

A = np.vstack([x,np.ones(n)]).T
print np.linalg.lstsq(A,y)[0]

