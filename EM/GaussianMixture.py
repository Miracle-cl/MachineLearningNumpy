import math
import numpy as np


class GaussianMixture:
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon
    
    @staticmethod
    def cal_phi(y, miu, sigma):
        const = 1 / math.sqrt(math.pi * 2)
        return const * math.exp(- math.pow(y - miu, 2) / (2 * math.pow(sigma, 2))) / sigma

    @staticmethod
    def update_alpha(gammas):
        return np.average(gammas, axis=0)

    @staticmethod
    def update_miu(gammas, Y):
        x1 = np.sum(gammas * Y, axis=0)
        x2 = np.sum(gammas, axis=0)
        return x1 / x2

    @staticmethod
    def update_sigma(gammas, mius, Y):
        x1 = np.sum(gammas * np.power(Y - mius, 2), axis=0)
        x2 = np.sum(gammas, axis=0)
        return np.sqrt(x1 / x2)

    # def init_theta(self, Y, K):
    #     return

    def e_step(self, Y, alphas, mius, sigmas):
        N = len(Y)
        K = len(alphas)
        gammas = np.zeros((N, K))
        
        for j in range(N):
            for k in range(K):
                gammas[j, k] = alphas[k] * self.cal_phi(Y[j], mius[k], sigmas[k])
        sum_ = np.sum(gammas, axis=1, keepdims=True)
        # print(sum_.shape)
        gammas /= sum_
        return gammas

    def m_step(self, gammas, mius, Y):
        mius_ = self.update_miu(gammas, Y)
        sigmas_ = self.update_sigma(gammas, mius, Y)
        alphas_ = self.update_alpha(gammas)
        return alphas_, mius_, sigmas_

    def fit(self, Y, alphas, mius, sigmas, max_iters=100):
        # ignore initial thetas
        Y = Y.reshape(len(Y), -1)
        for _ in range(max_iters):
            gammas = self.e_step(Y, alphas, mius, sigmas)
            alphas, mius, sigmas = self.m_step(gammas, mius, Y)
            print(alphas, mius, sigmas)
            if abs(mius[0] - mius[1]) < self.epsilon:
                break
        return alphas, mius, sigmas


if __name__ == '__main__':
    Y = np.array([-67,-48,6,8,14,16,23,24,28,29,41,49,56,60,75])
    # K = 2
    alphas = np.array([0.5, 0.5])
    mius = np.array([-1, 1])
    sigmas = np.array([100, 100])
    
    gmm = GaussianMixture()
    thetas = gmm.fit(Y, alphas, mius, sigmas)
    print(thetas)