import math
import numpy as np
from scipy import stats
from collections import Counter

class EM:
    @staticmethod
    def two_coin_model():
        # expectation maximization algorithm: 2 coins model
        # theta_a :  A硬币正面的概率
        # theta_b :  B硬币正面的概率
        # pa :  当前硬币是A的概率
        # pb :  当前硬币是B的概率
        n1n2s = [(5,5), (9,1), (8,2), (4,6), (7,3)] # [H, T]
        theta_a, theta_b = 0.6, 0.5 # initial value
        epochs = 10

        for epoch in range(epochs):
            # E step
            a_H = a_T = 0
            b_H = b_T = 0
            for n1, n2 in n1n2s:
                pa = math.pow(theta_a, n1) * math.pow(1-theta_a, n2)
                pb = math.pow(theta_b, n1) * math.pow(1-theta_b, n2)
                # normalization
                pa, pb = pa / (pa + pb), pb / (pa + pb)
                a_H += pa * n1
                a_T += pa * n2
                b_H += pb * n1
                b_T += pb * n2
            # M step
            theta_a = a_H / (a_H + a_T)
            theta_b = b_H / (b_H + b_T)

            print(epoch, theta_a, theta_b)
        return theta_a, theta_b

    @staticmethod
    def three_coin_model():
        observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                                [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                                [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

        a, b, c = [0.5, 0.8, 0.6] # initial probability
        n = observations.shape[1]
        epochs = 10
        for epoch in range(epochs):
            cnt = Counter()
            c_probs = []
            # E step
            for ob in observations:
                n1 = sum(ob) # '1' is up
                n2 = n - n1 # '0' is up
                # ca = c * math.pow(a, n1) * math.pow(1-a, n2)
                # cb = (1-c) * math.pow(b, n1) * math.pow(1-b, n2)
                # same as befor row
                ca = c * stats.binom.pmf(n1, n, a)
                cb = (1-c) * stats.binom.pmf(n1, n, b)
                ca, cb = ca / (ca + cb), cb / (ca + cb) # normalization
                c_probs.append(ca)
                cnt['AH'] += ca * n1
                cnt['AT'] += ca * n2
                cnt['BH'] += cb * n1
                cnt['BT'] += cb * n2

            # M step
            c = np.mean(c_probs)
            a = cnt['AH'] / (cnt['AH'] + cnt['AT'])
            b = cnt['BH'] / (cnt['BH'] + cnt['BT'])
            print(epoch, a, b, c)
        return a, b, c

    @staticmethod
    def homework_91():
        pi = 0.46
        p = 0.55
        q = 0.67
        Y = np.array([1,1,0,1,0,0,1,0,1,1])

        def cal_miu(y, pi, p, q):
            pb = pi * math.pow(p, y) * math.pow(1-p, 1-y)
            pc = (1 - pi) * math.pow(q, y) * math.pow(1-q, 1-y)
            return pb / (pb + pc)

        def gen_para(Y, pi, p, q):
            MIU = np.array([cal_miu(y, pi, p, q) for y in Y])
            pi = np.mean(MIU)
            p = MIU.dot(Y) / np.sum(MIU)
            q = (1 - MIU).dot(Y) / np.sum(1-MIU)
            return pi, p, q

        for _ in range(10):
            pi, p, q = gen_para(Y, pi, p, q)
            print(pi, p, q)


if __name__ == '__main__':
    em = EM()
    print(em.two_coin_model())