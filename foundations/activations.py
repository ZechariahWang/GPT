import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        new_list=[]
        for i in range(len(z)):
            item=1 / (1 + np.exp(-z[i]))
            new_list.append(item)

        return np.round(np.array(new_list,dtype=np.float64),5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        new_list=[]
        for i in range(len(z)):
            item=max(0,z[i])
            new_list.append(item)

        return np.round(np.array(new_list,dtype=np.float64),5)

