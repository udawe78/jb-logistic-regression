# This models make predictions of the cancer tumor type (malignant or benign)
# based on three diagnostic parameters: 'worst concave points', 'worst perimeter', 'worst radius'.
# The models are trained on the Breast cancer wisconsin (diagnostic) dataset by gradient descent.


import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.weights = []

    def predict_proba(self, row, coef_):
        t = np.dot(row, coef_)
        proba = 1 / (1 + math.exp(-t))
        return proba

    def fit_mse(self, x_train, y_train):
        x_train = x_train.copy()

        if self.fit_intercept:
            x_train.insert(0, 'x_0', 1.0)

        # initialize weights
        weights = [0 for _ in range(x_train.shape[1])]

        # fit the model
        for num_of_epoch in range(self.n_epoch):
            for i, row in x_train.iterrows():

                y_hat = self.predict_proba(row, weights)

                # update all weights
                for j, x in enumerate(row):
                    weights[j] -= self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat) * x

        self.weights = weights  # redefine weights

    def fit_log_loss(self, x_train, y_train):
        x_train = x_train.copy()

        if self.fit_intercept:
            x_train.insert(0, 'x_0', 1.0)

        # initialized weights
        weights = [0 for _ in range(x_train.shape[1])]

        n = x_train.shape[0]

        for num_of_epoch in range(self.n_epoch):
            for i, row in x_train.iterrows():

                y_hat = self.predict_proba(row, weights)

                # update all weights
                for j, x in enumerate(row):
                    weights[j] -= self.l_rate * (y_hat - y_train[i]) * x / n

        self.weights = weights  # redefine weights

    def predict(self, x_test, cut_off=0.5):
        x_test = x_test.copy()

        if self.fit_intercept:
            x_test.insert(0, 'x_0', 1.0)

        predict = []
        for i, row in x_test.iterrows():
            y_hat = self.predict_proba(row, self.weights)
            if y_hat < cut_off:
                predict.append(0)
            else:
                predict.append(1)

        return predict  # predictions are binary values - 0 or 1


# input data
used_params = ['worst concave points', 'worst perimeter', 'worst radius']
params_values = [float(input(f"Enter the value of the parameter '{i}': ")) for i in used_params]
params_values = dict(zip(used_params, params_values))
params_values = pd.DataFrame(data=params_values, index=[0])
print('\nUSED PARAMETERS:', params_values, sep='\n')

# load training dataset
data = load_breast_cancer(as_frame=True)
X = pd.DataFrame(data['data'], columns=used_params)
Y = data['target']

# standardize X
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# split train-test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=43)

# create the class instances
print('\nPlease, wait while the result is predicted...')
clr = CustomLogisticRegression()
lr = LogisticRegression(max_iter=100)

# output lists
accuracies = []
predicts = []

# fit the custom model via mse method
clr.fit_mse(X_train, Y_train)
accuracies.append(round(accuracy_score(Y_test, clr.predict(X_test)), 3))
predicts.append(clr.predict(params_values))

# fit the custom model via log_loss method
clr.fit_log_loss(X_train, Y_train)
accuracies.append(round(accuracy_score(Y_test, clr.predict(X_test)), 3))
predicts.append(clr.predict(params_values))

# fit the sklearn model
lr.fit(X_train, Y_train)
accuracies.append(round(lr.score(X_test, Y_test), 3))
predicts.append(lr.predict(params_values))

# output the result as Dataframe
result = {'predict': predicts, 'accuracy': accuracies}
result = pd.DataFrame(data=result, index=['mse', 'log-loss', 'sklearn'])

print('\nRESULT:\n', result)
print('\nwhere: [0] - MALIGNANT, [1] - BENIGN')
