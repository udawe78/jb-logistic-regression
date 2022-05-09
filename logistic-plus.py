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
        self.coef_ = []
        self.mse_error_first = []  # mean square error (MSE) during the first epoch of learning (EOL)
        self.mse_error_last = []  # MSE during the last EOL
        self.log_loss_error_first = []  # log-loss function  during the first EOL
        self.log_loss_error_last = []  # log-loss function  during the last EOL

    def sigmoid(self, t):
        return 1 / (1 + math.exp(-t))

    def predict_proba(self, row, coef_):
        t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        x_train = X_train.copy()

        if self.fit_intercept:
            x_train.insert(0, 'x_0', 1.0)

        # initialize weights
        coef_ = [0 for _ in range(x_train.shape[1])]

        n = x_train.shape[0]

        # fit the model
        for num_of_epoch in range(self.n_epoch):
            for i, row in x_train.iterrows():

                y_hat = self.predict_proba(row, coef_)

                if num_of_epoch == 0:
                    self.mse_error_first.append(((y_hat - y_train[i]) ** 2) / n)

                if num_of_epoch == self.n_epoch - 1:
                    self.mse_error_last.append(((y_hat - y_train[i]) ** 2) / n)

                # update all weights
                for j, x in enumerate(row):
                    coef_[j] -= self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat) * x

        # redefine weights
        self.coef_ = coef_

    def fit_log_loss(self, X_train, y_train):
        x_train = X_train.copy()

        if self.fit_intercept:
            x_train.insert(0, 'x_0', 1.0)

        # initialized weights
        coef_ = [0 for _ in range(x_train.shape[1])]

        n = x_train.shape[0]

        for num_of_epoch in range(self.n_epoch):
            for i, row in x_train.iterrows():

                y_hat = self.predict_proba(row, coef_)

                if num_of_epoch == 0:
                    self.log_loss_error_first.append((y_train[i] * math.log(y_hat) +
                                                     (1 - y_train[i]) * math.log(1 - y_hat)) / -n)

                if num_of_epoch == self.n_epoch - 1:
                    self.log_loss_error_last.append((y_train[i] * math.log(y_hat) +
                                                    (1 - y_train[i]) * math.log(1 - y_hat)) / -n)

                # update all weights
                for j, x in enumerate(row):
                    coef_[j] -= self.l_rate * (y_hat - y_train[i]) * x / n

        # redefine weights
        self.coef_ = coef_

    def predict(self, X_test, cut_off=0.5):
        x_test = X_test.copy()

        if self.fit_intercept:
            x_test.insert(0, 'x_0', 1.0)

        y_pred = []
        for i, row in x_test.iterrows():
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat < cut_off:
                y_pred.append(0)
            else:
                y_pred.append(1)

        return y_pred  # predictions are binary values - 0 or 1


# input data
data = load_breast_cancer(as_frame=True)
X = pd.DataFrame(data['data'], columns=['worst concave points', 'worst perimeter', 'worst radius'])
y = data['target']

# standardize X
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

# create the class instances
clr = CustomLogisticRegression(n_epoch=1000)
lr = LogisticRegression(max_iter=1000)

# fit the custom model via mse method
clr.fit_mse(X_train, y_train)
mse_y_pred = clr.predict(X_test)
mse_accuracy = accuracy_score(y_test, mse_y_pred)

# fit the custom model via log_loss method
clr.fit_log_loss(X_train, y_train)
log_loss_error_first = clr.log_loss_error_first
log_loss_error_last = clr.log_loss_error_last
log_loss_y_pred = clr.predict(X_test)
log_loss_accuracy = accuracy_score(y_test, log_loss_y_pred)

# fit the sklearn model
lr.fit(X_train, y_train)
sklearn_accuracy = lr.score(X_test, y_test)

# output the result
answer_dict = {'mse_accuracy': mse_accuracy,
               'logloss_accuracy': log_loss_accuracy,
               'sklearn_accuracy': sklearn_accuracy,
               'mse_error_first': mse_error_first,
               'mse_error_last': mse_error_last,
               'logloss_error_first': log_loss_error_first,
               'logloss_error_last': log_loss_error_last}

print(answer_dict)

print(f'''answers to the questions:
1) {round(min(mse_error_first), 5)}
2) {round(min(mse_error_last), 5)}
3) {round(max(log_loss_error_first), 5)}
4) {round(max(log_loss_error_last), 5)}
5) expanded
6) expanded''')
