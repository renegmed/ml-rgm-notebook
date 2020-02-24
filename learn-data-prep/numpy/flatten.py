'''
‘C’ means to flatten in row-major (C-style) order. 
‘F’ means to flatten in column-major (Fortran- style) order. 
‘A’ means to flatten in column-major order if a is Fortran 
contiguous in memory, row-major order otherwise. 
‘K’ means to flatten a in the order the elements occur in memory. The default is ‘C’.
'''

import numpy as np
y = np.array([[1, 2],[3, 4]])
print(y)

x =  y.flatten()
print("1. y.flatten()\n {}".format(x))

x =  y.flatten('F')
print("2. y.flatten('F')\n {}".format(x))

print('\n\n')
w = np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]],[[9,10],[11,12]]])
print(w)

z = w.flatten()
print("1.  w.flatten()\n {}".format(z))

z = w.flatten('F')
print("2.  w.flatten('F')\n {}".format(z))

