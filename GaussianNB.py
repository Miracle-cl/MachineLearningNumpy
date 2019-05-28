import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from math import pi, pow, sqrt, exp


def create_data():
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:, :])
    # print(data)
    return data[:,:-1], data[:,-1]

class MyGaussianNB:
    def __init__(self):
        # data_describe is a dict to record
        # mean and std of every feature when label=y
        # key is label : int
        # value is mean and std of every feature when label : np.array.shape(features x 2)
        self.data_description = dict()

    def fit(self, x_train, y_train):
        # calculate mean and std
        # x_train-shape: n_samples x features
        # y_train-shape: n_samples x classes
        labels = set(y_train)
        for label in labels:
            l_idx = np.where(y_train == label) # label=1: idx of label=1
            means = np.mean(x_train[l_idx], axis=0)
            stds = np.std(x_train[l_idx], axis=0)
            self.data_description[label] = np.stack((means, stds), axis=-1)
        return "GussianNB train done!"

    def probability_density(self, xi, mean, std):
        # Gaussian/Normal Distribution: Probability Density Function  
        # probability = probability_density * dx, same dx
        return exp(- 0.5 * pow((xi - mean) / std, 2)) / (std * sqrt(2 * pi))

    def calculate_probs(self, input_data):
        # input_data : features array of 1 sample [.....]
        probs = dict() # key-label; value-probs_density
        for label, description in self.data_description.items():
            probs[label] = 1 # prob of (y = ck) if classes are uniform then get 1 else num_class / all_class 
            for i, xi in enumerate(input_data):
                mean, std = description[i]
                probs[label] *= self.probability_density(xi, mean, std)
        return probs

    def predict(self, input_data):
        # predict label of input_data
        return sorted(self.calculate_probs(input_data).items(), key=lambda x: x[1], reverse=True)[0][0]

    def score(self, x_test, y_test):
        # get score of x_test, y_test
        pred = np.array([self.predict(x) for x in x_test])
        return np.sum(pred == y_test) / len(y_test)

def main():
    # create data
    X, y = create_data()
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nb = MyGaussianNB()
    nb.fit(X_train, y_train)
    print("clf score: ", nb.score(X_test, y_test))
    print("probs_density of x[0]: ", nb.calculate_probs(X_test[0]))
    ## after checking, result is same as sklearn.naive_bayes.GaussianNB

if __name__ == "__main__":
    main()
