#9th
import numpy as np
import matplotlib.pyplot as p
def local_regression(x0, X, Y, tau):
    x0 = [1, x0]   
    X = [[1, i] for i in X]
    X = np.asarray(X)
    xw = (X.T) * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau))
    beta = np.linalg.pinv(xw @ X) @ xw @ Y @ x0  
    return beta    
def draw(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    p.plot(X, Y, 'o', color='black')
    p.plot(domain, prediction, color='red')
    p.show()
X = np.linspace(-3, 3, num=1000)
domain = X
Y = np.log(np.abs(X ** 2 - 1) + .5)
draw(10)
draw(0.1)
draw(0.01)
draw(0.001)
