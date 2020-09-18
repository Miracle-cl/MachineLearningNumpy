# reference: https://datawhalechina.github.io/statistical-learning-method-solutions-manual/#/chapter5/chapter5

import numpy as np


class CART:
    def __init__(self, epsilon):
        self.eps = epsilon
        
    def fit(self, x, y):
        self.tree = self._fit(x, y)
        
    def _fit(self, x, y):
        # choose j & s to split data
        j, s, min_cost, c1, c2 = self._divide(x, y)
        val = x[s, j]
        left_len = len(y[np.where(x[:, j] <= val)])
        if min_cost < self.eps or left_len <= 1:
            left = c1
        else:
            left = self._fit(x[np.where(x[:, j] <= val)], 
                             y[np.where(x[:, j] <= val)])
        
        right_len = len(y) - left_len
        if min_cost < self.eps or right_len <= 1:
            right = c2
        else:
            right = self._fit(x[np.where(x[:, j] > val)], 
                              y[np.where(x[:, j] > val)])
        tree = {'feature': j, 'value': val, 'left': left, 'right': right}
        return tree         

    @staticmethod
    def _divide(x, y):
        n_sample, n_feature = x.shape
        min_cost = float('inf')
        for i in range(n_feature):
            for k in range(n_sample):
                value = x[k, i]
                # split left and right part by value
                y1 = y[np.where(x[:, i] <= value)]
                c1 = np.mean(y1) if y1.size else 0
                y1 = y1 - c1
                y2 = y[np.where(x[:, i] > value)]
                c2 = np.mean(y2) if y2.size else 0
                y2 = y2 - c2
                cost = y1.dot(y1) + y2.dot(y2)
                if cost < min_cost:
                    min_cost = cost
                    j, s, mc1, mc2 = i, k, c1, c2

        return j, s, min_cost, mc1, mc2
    
    
if __name__ == '__main__':
    from pprint import pprint
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
    y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])
    cart = CART(epsilon=0.2)
    cart.fit(x, y)
    pprint(cart.tree)