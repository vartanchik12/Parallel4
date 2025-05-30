import numpy as np

A = np.genfromtxt('matrixA.txt', delimiter=' ')
A = A.astype('int32')

B = np.genfromtxt('matrixB.txt', delimiter=' ')
B = B.astype('int32')

PY_res=np.dot(A,B)

CPP_res = np.genfromtxt('matrix_result.txt', delimiter=' ')
CPP_res = CPP_res.astype('int32')

if np.array_equal(PY_res, CPP_res):
    print("Correct")
else:
    print("Not correct")