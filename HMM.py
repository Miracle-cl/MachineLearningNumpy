import numpy as np
from pprint import pprint


class HiddenMarkov_V2:
    # Q : set of state
    # V : set of observation
    # A : matrix of probability of state transfer - len(Q) * len(Q)
    # B : matrix of probability of observation - len(Q) * len(V)
    # O : observation sequence
    # PI : vector of probability of original state - len(PI) == len(Q)
    @staticmethod
    def viterbi(V, A, B, O, PI):
        N = len(A) # num of states
        T = len(O) # num of observations
        prob_mat = np.zeros((N, T))
        path_mat = np.zeros((N, T)).astype(int)
        # cal first state by first observation
        color = V.index(O[0])
        states = B[:, color] * PI
        prob_mat[:, 0] = states
        # path_mat[:, 1] = np.argmax(nxt, axis=0)
        for t in range(1, T):
            states = states.reshape(3,1)
            color = V.index(O[t])
            nxt = states * A * B[:, color]
            states = np.max(nxt, axis=0)
            prob_mat[:, t] = states
            path_mat[:, t] = np.argmax(nxt, axis=0)
        # pprint(prob_mat)
        # pprint(path_mat)

        last_prob = np.max(prob_mat[:, T-1])
        last_state = np.argmax(prob_mat[:, T-1])
        hid_states = [last_state]
        for t in range(T-1, 0, -1):
            s = hid_states[-1]
            # path_mat[s, t] is the former state
            hid_states.append(path_mat[s, t])
        hid_states.reverse()
        return [s+1 for s in hid_states], last_prob


class HiddenMarkov:
    # Q : set of state
    # V : set of observation
    # A : matrix of probability of state transfer - len(Q) * len(Q)
    # B : matrix of probability of observation - len(Q) * len(V)
    # O : observation sequence
    # PI : vector of probability of original state - len(PI) == len(Q)
    def forward(self, Q, V, A, B, O, PI):  # forward algorithms
        N = len(Q)  # length of state seq
        M = len(O)  # length of observation seq
        T = M  # time steps == length of observation seq
        alphas = np.zeros((N, T))  # alpha matrix

        for t in range(T):
            index_Ot = V.index( O[t] )
            if t == 0:
                alphas[:, t] = PI * B[:, index_Ot]
            else:
                pi_t = np.dot(alphas[:, t-1].reshape(1, -1), A).reshape(-1) # P176-10.16 [  ]
                alphas[:, t] = pi_t * B[:, index_Ot]
        prob_O = np.sum(alphas[:, T-1]) # P176-10.17

        return alphas, prob_O

    def backward(self, Q, V, A, B, O, PI):  # backward algorithms
        N = len(Q)  # length of state seq
        M = len(O)  # length of observation seq
        T = M  # time steps == length of observation seq
        betas = np.ones((N, T + 1))  # beta matrix and initialize

        for t in range(T-1, 0, -1):
            index_Ot = V.index( O[t] )
            # for i in range(N):
                # betas[i, t] = np.sum(A[i, :] * B[:, index_Ot] * betas[:, t+1])
            betas[:, t] = A.dot( B[:, index_Ot] * betas[:, t+1] )  ## same as for loop before

        prob_O = np.sum(betas[:, 1] * PI * B[:, V.index( O[0] )])
        return betas[:, 1:], prob_O

    def viterbi(self, V, A, B, O, PI):
        N = len(A)  # length of state seq
        T = len(O)  # length of observation seq

        deltas = np.zeros((N, T))  # delta matrix and initialize
        path_matrix = np.zeros((N, T), dtype=np.int)

        for t in range(T):
            index_Ot = V.index( O[t] )
            if t == 0:
                deltas[:, t] = PI * B[:, index_Ot]
            else:
                for j in range(N):
                    v = deltas[:, t-1] * A[:, j]
                    deltas[j, t] = np.max(v) * B[j, index_Ot]
                    path_matrix[j, t] = np.argmax(v)
        prob_best_path = np.max(deltas[:, T-1])
        I_best = []
        last_state = np.argmax(deltas[:, T-1])
        I_best.append(last_state)
        for step in range(T-2, -1, -1):
            I_best.append(path_matrix[I_best[-1], step+1])
        I_best.reverse()
        # print(prob_best_path)
        return [s+1 for s in I_best], prob_best_path


if __name__ == "__main__":
    # variables
    Q = [1, 2, 3]
    V = ['red', 'white']
    O = ['red', 'white', 'red', 'white']
    PI = np.array([0.2, 0.4, 0.4])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    A = np.array([[.5, .2, .3],
                  [.3, .5, .2],
                  [.2, .3, .5]])

    hmm = HiddenMarkov()
    # _, p1 = hmm.forward(Q, V, A, B, O, PI)
    # _, p2 = hmm.backward(Q, V, A, B, O, PI)
    best_state_path, prob = hmm.viterbi(V, A, B, O, PI) # state + 1
    print(best_state_path, prob)
