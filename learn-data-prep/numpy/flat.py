'''
The ndarray.flat() function is used to make 1-D iterator over 
the array.

This is a numpy.flatiter instance, which acts similarly to, 
but is not a subclass of, Python’s built-in iterator object.
'''

import numpy as np

y = np.arange(2, 8)
print("1. np.arange(2, 8)\n{}\n".format(y))
y = y.reshape(3, 2)
print("2. y.reshape(3, 2)\n{}\n".format(y))

z = y.flat[4]
print("3. y.flat[4]\n{}\n".format(z))

y.flat = 4
print("4. y.flat = 4\n{}\n".format(y))

y.flat[[2, 5]] = 2
print("5. y.flat[[2, 5]] = 2\n{}\n".format(y))

y.flat[[0, 3, 4]] = 7
print("6. y.flat[[0, 3, 4]] = 7\n{}".format(y))