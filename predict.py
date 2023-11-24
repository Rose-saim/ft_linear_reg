import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearReg import MyLinearRegression 



def mileage_predict(theta, mileage):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if np.shape(theta) != (2,):
    # #and not mileage.type(int, float):
        return 'Error'
    #Store the current value of theta[0]
    theta_0_values = theta[0] + (theta[1] * mileage)
    # Update theta[0]
    return theta_0_values


def train_mileage(price, mil, theta):
    # if theta.size() is not mileage.size():
    #     return None
    mileage = mil
    price = price
    m = len(mil)
    learnRate = 0.1
    estimate_p = mileage_predict(theta, mileage)
    err = estimate_p - price
    print(err)
    t0  = np.sum(err)
    t1  = np.sum(err * mileage)
    #print(result, learnRate, m)
    theta0 = (t0 * learnRate)/m
    #print(theta1)
    theta1 = (t1 * learnRate)/m
    return theta0, theta1



def main():
    #read_csv
    data = pd.read_csv('./data.csv')
    #init theta to 0
    theta = [0, 0]
    theta[0] = 0
    theta[1] = 0
    a = MyLinearRegression(theta)
    X = data['km'].values
    Y = data['price'].values
    Xnorm, x_mean, x_std = a.norm_value(X, X.min(), X.max())
    a.fit_(Xnorm, Y)
    a.denorm_value(x_mean, x_std)
    y_hat = a.predict_(X)
    plt.plot(a.l_err, label='Cost error')
    plt.xlabel('kilometers')
    plt.ylabel('prices')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.plot(a.mse_v, label='mse')
    plt.title('Graphe mse')
    plt.xlabel('kilometers')
    plt.ylabel('prices')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.scatter(X, Y)
    plt.plot(X, y_hat, color='red', label='linear regression')

    plt.xlabel('kilometers')
    plt.ylabel('prices')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()