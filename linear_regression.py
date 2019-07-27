import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score

import visualize as vs


def linear_regression(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test):

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    # Plot the model
    vs.scatter_for_regression(diabetes_X_test, diabetes_y_test, regr.predict(diabetes_X_test), 'blue')



def my_linear_regression(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test):
    w0 = np.random.rand()
    w1 = np.random.rand()

    X = diabetes_X_train
    y = diabetes_y_train

    steps = 1000
    alpha = 0.5

    for i in range(0, steps) :
        w0, w1 = updateParameters(w0, w1, X, y, alpha)

    rss = np.mean((w0 + w1 * diabetes_X_test - diabetes_y_test) ** 2)
    print('My Residual sum of squares: %.2f' % rss)
    print('My R2 score: %.2f'% r2_score(diabetes_y_test, w0 + w1 * diabetes_X_test))
    vs.scatter_for_regression(diabetes_X_test, diabetes_y_test, f(diabetes_X_test, w0, w1), 'red')


def my_linear_regression_with_ridge(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test):
    w0 = np.random.rand()
    w1 = np.random.rand()

    X = diabetes_X_train
    y = diabetes_y_train

    steps = 1000
    alpha = 0.5

    for i in range(0, steps) :
        w0, w1 = updateParametersWithRidge(w0, w1, X, y, alpha)

    rss = np.mean((w0 + w1 * diabetes_X_test - diabetes_y_test) ** 2)
    print('My Residual sum of squares (ridge): %.2f' % rss)
    print('My R2 score (ridge): %.2f'% r2_score(diabetes_y_test, w0 + w1 * diabetes_X_test))
    vs.scatter_for_regression(diabetes_X_test, diabetes_y_test, f(diabetes_X_test, w0, w1), 'green')


def f(X, w0, w1) :
    return w0 + w1 * X



def updateParameters(w0, w1, X, y, alpha) :
    pred = f(X, w0, w1)

    t0 = w0 + 2 * alpha * (y - pred).mean()
    t1 = w1 + 2 * alpha * (np.dot(X.transpose(), y[:, np.newaxis] - pred))

    return t0, t1



def updateParametersWithRidge(w0, w1, X, y, alpha) :
    pred = f(X, w0, w1)

    t0 = w0 + 2 * alpha * (y - pred).mean() - 4 * alpha * w0 / len(X)
    t1 = w1 + 2 * alpha * (np.dot(X.transpose(), y[:, np.newaxis] - pred)) - 4 * alpha * w1 / len(X)

    return t0, t1



def derivatives(X, pred, y) :
    dW0 = (pred - y).mean()
    dW1 = ((pred - y) * X.reshape(-1, 1)).mean()

    return dW0, dW1



if __name__ == '__main__':
    #load dataset
    dataset = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = dataset.data[:, np.newaxis]
    diabetes_X_temp = diabetes_X[:, :, 0]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X_temp[:-20]
    diabetes_X_test = diabetes_X_temp[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = dataset.target[:-20]
    diabetes_y_test = dataset.target[-20:]

    # do simple linear regression
    linear_regression(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test)

    # do my linear regression
    my_linear_regression(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test)

    # do my linear regression with Ridge
    my_linear_regression_with_ridge(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test)