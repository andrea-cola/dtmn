import numpy as np
from sklearn import datasets, linear_model

import visualize as vs


def linear_regression(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test):

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # The coefficients
    print('Coefficients: \n', regr.coef_)

    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    # Plot the model
    vs.scatter_for_regression(diabetes_X_test, diabetes_y_test, regr.predict(diabetes_X_test))



def my_linear_regression(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test):
    w0 = np.random.rand()
    w1 = np.random.rand()

    X = diabetes_X_train
    y = diabetes_y_train

    steps = 1000
    alpha = 0.005

    for i in range(0, steps) :
        w0, w1 = updateParameters(w0, w1, X, y, alpha)

    print("MyRSS: ", np.mean((w0 + w1 * diabetes_X_test - diabetes_y_test) ** 2))



def updateParameters(w0, w1, X, y, alpha) :
    dW0, dW1 = derivatives(w0, w1, X, y)
    w0 = w0 - (alpha * dW0)
    w1 = w1 - (alpha * dW1)

    return w0, w1



def derivatives(w0, w1, X, y) :
    dW0 = 0
    dW1 = 0

    for (xi, yi) in zip(X, y) :
        dW0 += w0 + w1 * xi - yi
        dW1 += (w0 + w1 * xi - yi) * xi

    dW0 = dW0 / len(X)
    dW1 = dW1 / len(X)

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
    my_linear_regression(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test)