
'''
  >> python3 reshape.py

  >> p3 reshape.py

  newshape:

  The new shape should be compatible with the original shape. If an integer, 
  then the result will be a 1-D array of that length. 
  
  One shape dimension can be -1. In this case, the value is inferred from 
  the length of the array and remaining dimensions.
'''

import numpy as np

y = np.array([[2,3,4], [5,6,7]])
print(y)

x = np.reshape(y, (3, 2))
print('0. {}\n{}'.format("np.reshape(y, (3, 2))",x))

x = np.reshape(y, (1, -1))
print('1. {}\n{}'.format("np.reshape(y, (1, -1))",x))

x = np.reshape(y, (3, -1))
print('2. {}\n{}'.format("np.reshape(y, (3, -1))",x))

x = np.reshape(y, (3,2) )
print('3. {}\n{}'.format("np.reshape(y, (3,2) )",x))

x = np.array([[2,3,4], [5,6,7]])
print('4. {}\n{}'.format("np.array([[2,3,4], [5,6,7]])",x))


x = np.reshape(y, 6)
print('5. {}\n{}'.format("np.reshape(y, 6)",x))


'''
  >> python3 reshape.py

  >> p3 reshape.py

  newshape:

  The new shape should be compatible with the original shape. If an integer, 
  then the result will be a 1-D array of that length. 
  
  One shape dimension can be -1. In this case, the value is inferred from 
  the length of the array and remaining dimensions.
'''

import numpy as np

y = np.array([[2,3,4], [5,6,7]])
print(y)

x = np.reshape(y, (3, 2))
print('0. {}\n{}'.format("np.reshape(y, (3, 2))",x))

x = np.reshape(y, (1, -1))
print('1. {}\n{}'.format("np.reshape(y, (1, -1))",x))

x = np.reshape(y, (3, -1))
print('2. {}\n{}'.format("np.reshape(y, (3, -1))",x))

x = np.reshape(y, (3,2) )
print('3. {}\n{}'.format("np.reshape(y, (3,2) )",x))

x = np.array([[2,3,4], [5,6,7]])
print('4. {}\n{}'.format("np.array([[2,3,4], [5,6,7]])",x))


x = np.reshape(y, 6)
print('5. {}\n{}'.format("np.reshape(y, 6)",x))


x = np.reshape(x, 6, order='F')
print('6. {}\n{}'.format("np.reshape(x, 6, order='F')",x))
