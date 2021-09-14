import numpy as np


class Math:
    def cantor_pairing(a, b):
        return int((0.5 * (a + b)*(a + b + 1)) + b)
    

    def mse(A, B):
        return np.mean(np.square(np.subtract(A, B)))
    

    def mae(A, B):
        return np.mean(np.abs(np.subtract(A, B)))