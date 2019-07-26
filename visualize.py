import matplotlib.pyplot as plt

def scatter_for_regression(x, y, y_predicted):
    plt.scatter(x, y, color='black')
    plt.plot(x, y_predicted, color='blue', linewidth=1)

    plt.show()
