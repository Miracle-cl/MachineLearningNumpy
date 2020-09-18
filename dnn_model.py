import numpy as np
from time import time

# Set how floating-point errors are handled: Take no action when the exception occurs
np.seterr(over='ignore')

# Architecture of DNN : 3 - 5 - 3 - 1
class NeuralNetwork():
    def __init__(self, l2=5, l3=3):
        np.random.seed(21)
        self.weights1 = 2 * np.random.random((3, l2)) - 1 #3*5
        self.weights2 = 2 * np.random.random((l2, l3)) - 1 #5*3
        self.weights3 = 2 * np.random.random((l3, 1)) - 1 #3*1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs_, labels_, training_iterations, learning_rate):
        for _ in range(training_iterations):
            # forward pass
            # inputs 4*3
            a2 = self.__sigmoid(np.dot(inputs_, self.weights1)) # 4*5
            a3 = self.__sigmoid(np.dot(a2, self.weights2)) # 4*3
            output = self.__sigmoid(np.dot(a3, self.weights3)) # 4*1

            # backwards pass
            # calculate dw3
            # derivative of MSE: (labels_ - output) * out * (1 - out) x a3
            # derivative of (binary)CrossEntropyLoss: (labels_ - output) x a3
            output_error = labels_ - output 
            output_error_term = output_error * self.__sigmoid_derivative(output) # 4*1
            delt_w3 = learning_rate * np.dot(a3.T, output_error_term) # 3*1 = 3*4 dot 4*1

            # calculate dw2
            h2_error = np.dot(self.weights3, output_error_term.T) # 3*4
            h2_error_term = h2_error * (self.__sigmoid_derivative(a3).T) # 3*4 = 3*4 * 3*4
            delt_w2 = learning_rate * np.dot(a2.T, h2_error_term.T) # 5*3 = 5*4 dot 4*3

            # calculate dw1
            h1_error = np.dot(self.weights2, h2_error_term) # 5*4 = 5*3 dot 3*4
            h1_error_term = h1_error * (self.__sigmoid_derivative(a2).T )# 5*4 = 5*4 * 5*4
            delt_w1 = learning_rate * np.dot(inputs_.T, h1_error_term.T) # 3*5 = 3*4 dot 4*5

            # update weights
            self.weights1 += delt_w1
            self.weights2 += delt_w2
            self.weights3 += delt_w3

    def forward_pass(self, inputs_):
        # forward pass
        a2 = self.__sigmoid(np.dot(inputs_, self.weights1))
        a3 = self.__sigmoid(np.dot(a2, self.weights2))
        output = self.__sigmoid(np.dot(a3, self.weights3))
        return output


if __name__ == "__main__":
    nn = NeuralNetwork()

    print("Initial weights:")
    print(nn.weights1)
    print(nn.weights2)
    print(nn.weights3)

    inputs_ = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    labels_ = np.array([0,1,1,0]).reshape((4,1))

    t0 = time()
    nn.train(inputs_, labels_, 10000, 1)
    print("\nTraining Time: {}s".format(time() - t0))

    print('\nAfter Training:')
    print(nn.weights1)
    print(nn.weights2)
    print(nn.weights3)
    print("\nInputs: [0,1,1]")
    print("\nOutputs: {}".format(nn.forward_pass(np.array([0,1,1]))))
