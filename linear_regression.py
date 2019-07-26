import numpy as np
from sklearn import datasets, linear_model

import visualize as vs


def simple_linear_regression(dataset):

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis]
    diabetes_X_temp = diabetes_X[:, :, 0]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X_temp[:-20]
    print(diabetes_X_train.shape)

    diabetes_X_test = diabetes_X_temp[-20:]
    print(diabetes_X_test.shape)

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    print(diabetes_y_train.shape)

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


if __name__ == '__main__':
    #load dataset
    diabetes = datasets.load_diabetes()

    # do simple linear regression
    simple_linear_regression(diabetes)