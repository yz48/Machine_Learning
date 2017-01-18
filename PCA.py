__author__ = 'yz48'
import numpy as np

X = np.random.randn(9, 6)
U, s, V = np.linalg.svd(X, full_matrices=False)
print U.shape, V.shape, s.shape
S = np.diag(s)

Mhat = np.dot(U, np.dot(S, V))
print np.allclose(X, Mhat)
Mhat2 = np.dot(U[:, :20], np.dot(S[:20, :20], V[:,:20]))

w, v = np.linalg.eig(np.dot(X.T,X))
print 'each row from svd represent eigenvector'
print V
print 'each column from eig represent eigenvector'
print v

print s*s
print w