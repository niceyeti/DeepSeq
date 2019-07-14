import numpy as np

#@x: An m x n matrix, whose rows are output probabilities
def logsumexp(x):
    a = x.max(axis=1).reshape(x.shape[0], 1) # get max of each row, reshape result to match @x
    print("a: "+str(a))
    z = x - a
    print("z: "+str(z))
    sumZ = np.sum(z, axis=1).reshape(x.shape[0], 1)
    print("sum z: "+str(sumZ))
    expZ = np.exp(z)
    print("expZ: "+str(expZ))
    return expZ / sumZ

m = 2
n = 3
"""
[[0,1,2],
 [3,4,5]]
"""
#x = np.random.rand(m,n)
x = np.linspace(1,m*n,m*n).reshape(m,n)



print("x: {}".format(x))
result = logsumexp(x)
print("result: {}".format(result))





