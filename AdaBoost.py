# reference: https://datawhalechina.github.io/statistical-learning-method-solutions-manual/#/chapter8/chapter8

import numpy as np

# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
# y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

x = np.array([[0, 1, 3],
              [0, 3, 1],
              [1, 2, 2],
              [1, 1, 3],
              [1, 2, 3],
              [0, 1, 2],
              [1, 1, 2],
              [1, 1, 1],
              [1, 3, 1],
              [0, 2, 1]])
y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])


class AdaBoost:
    def __init__(self, x, y, eps=0.01):
        self.x = x
        self.y = y
        self.n_sample, self.n_feature = self.x.shape
        self.w = np.full((self.n_sample,), 1 / self.n_sample)
        self.rules = ['left', 'right']
        self.G = []
        self.eps = eps

    def base_estimator(self, dim, threshVal, rule):
        res = np.ones((self.n_sample, ))
        if rule == 'left':
            # left = -1, right = 1
            res[np.where(self.x[:, dim] <= threshVal)] = -1
        if rule == 'right':
            # left = 1, right = -1
            res[np.where(self.x[:, dim] > threshVal)] = -1
        return res

    def build_stump(self):
        min_err = float('inf')
        para = None
        # split data x by val in x - dim
        for dim in range(self.n_feature):
            xd = self.x[:, dim]
            x_vals = list(set(xd))
            x_vals.sort()
            for val in x_vals:
                for rule in self.rules:
                    pred = self.base_estimator(dim=dim, threshVal=val, rule=rule)
                    error = (pred != self.y).dot(self.w)
                    if error < min_err:
                        min_err = error
                        para = (dim, val, rule)
        return min_err, para

    def update_w(self, alpha, pred):
        self.w *= np.exp(-alpha * self.y * pred)
        Z = self.w.sum()
        self.w = self.w / Z
        
    def fit(self, iters=5):
        fx_pred = np.zeros((self.n_sample,))
        for it in range(iters):
            # build Gm and cal alpha_m
            min_err, g_para = self.build_stump()
            alpha = 0.5 * np.log((1-min_err) / min_err)
            # print((min_err, alpha, g_para))
            self.G.append((alpha, g_para))
            
            # update weight of samples
            g_pred = self.base_estimator(*g_para)
            self.update_w(alpha, g_pred)
            
            # linear combination of Gm: sum(alpha * Gm)
            fx_pred += alpha * g_pred
            
            # uncorrect classification in sum(alpha * Gm)
            uncorrs = np.sum(np.sign(fx_pred) != self.y)
            uncorr_rate = uncorrs / self.n_sample
            if uncorr_rate < self.eps:
                print(f'Finished with {it+1} Iters.')
                print(np.sign(fx_pred))
                break



ab = AdaBoost(x, y)
ab.fit(iters=20)
print(ab.G)