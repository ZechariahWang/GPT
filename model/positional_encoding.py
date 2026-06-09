import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.

        # arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        # auto_matrix = arr.reshape(2, -1)
        # print(auto_matrix)
        # Output:
        # [[1 2 3 4]
        # [5 6 7 8]]

        # arr = np.arange(3, 7)
        # print(arr) 
        # Output: [3 4 5 6]

        PE=np.zeros((seq_len, d_model))
        position = np.arange(seq_len).reshape(-1, 1)
        div = 10000 ** (np.arange(0,d_model,2) / d_model)
        PE[:, 0::2] = np.sin(position / div)
        PE[:, 1::2] = np.cos(position / div)

        return np.round(PE, decimals=5)