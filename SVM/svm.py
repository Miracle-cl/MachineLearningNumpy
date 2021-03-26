# Reference : https://github.com/LasseRegin/SVM-w-SMO/blob/master/SVM.py

import numpy as np
import random as rnd
# from sklearn.metrics import accuracy_score

rnd.seed(100)

class SVM():
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Initialization
        n = X.shape[0]
        alpha = np.zeros(n)
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(n):
                i = self.get_rnd_int(0, n-1, j)
                xi, xj, yi, yj = X[i,:], X[j,:], y[i], y[j]
                k = kernel(xi, xi) + kernel(xj, xj) - 2 * kernel(xi, xj)
                if k == 0:
                    continue
                pre_aj, pre_ai = alpha[j], alpha[i]
                L, H = self.compute_L_H(self.C, pre_aj, pre_ai, yj, yi)
                
                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)
                
                # Compute E_i, E_j
                E_i = self.E(xi, yi, self.w, self.b)
                E_j = self.E(xj, yj, self.w, self.b)

                # Set new alpha values
                alpha[j] = pre_aj + float(yj * (E_i - E_j)) / k
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = pre_ai + yi * yj * (pre_aj - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return None, count

        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count

    def predict(self, X):
        return self.h(X, self.w, self.b)

    @staticmethod
    def calc_b(X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    @staticmethod
    def calc_w(alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))

    @staticmethod
    def h(X, w, b):
        # Prediction
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    @classmethod
    def E(cls, x_k, y_k, w, b):
        # Prediction error
        return cls.h(x_k, w, b) - y_k

    @staticmethod
    def compute_L_H(C, pre_aj, pre_ai, yj, yi):
        if yi != yj:
            L, H = max(0, pre_aj - pre_ai), min(C, C - pre_ai + pre_aj)
        else:
            L, H = max(0, pre_ai + pre_aj - C), min(C, pre_ai + pre_aj)
        return L, H

    @staticmethod
    def get_rnd_int(a, b, z):
        # get randint between a and b but is different with z
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt += 1
        return i

    # Define kernels
    @staticmethod
    def kernel_linear(x1, x2):
        return np.dot(x1, x2.T)

    @staticmethod
    def kernel_quadratic(x1, x2):
        return np.dot(x1, x2.T) ** 2



if __name__ == '__main__':
    path = 'MachineLearningNumpy/SVM/iris-slwc.txt'
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append([float(s) for s in line.strip().split(',')])
    
    data = np.array(data)
    X, Y = data[:, :2], data[:, 2]
    Y = Y.astype(int)

    svm = SVM()
    support_vectors, count = svm.fit(X, Y)
    y_pred = svm.predict(X)
    # score = accuracy_score(Y, y_pred)
    score = (Y == y_pred).sum() / len(Y)
    print([count, score])
    print(support_vectors)
    print(svm.w)
    print(svm.b)
