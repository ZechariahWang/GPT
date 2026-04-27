import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        result=[]

        total_sum=0
        for i in range(len(z)):
            total_sum += np.exp(z[i] - max(z))
        
        for i in range (len(z)):
            softmax = np.exp(z[i]-max(z)) / total_sum
            result.append(softmax)

        return np.round(np.array(result, dtype=np.float64), 4)

        
