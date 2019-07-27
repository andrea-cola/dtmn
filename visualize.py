import matplotlib.pyplot as plt

def scatter_for_regression(x, y, y_predicted, color):
    plt.scatter(x, y, color='black')
    plt.plot(x, y_predicted, color=color, linewidth=1)

    plt.show()
