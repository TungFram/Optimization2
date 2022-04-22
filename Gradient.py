import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from warnings import warn
from sklearn.datasets import make_spd_matrix
from numpy.linalg import norm
from decimal import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import *
from numpy import *
from sympy import symbols, Eq, solve
from math import sqrt
# номер функции соответствует номеру частной производная
# 1/3 x^2 + 1/3 y^2 +1/5 xy

def f(x,y):
    z = 2*x*x+((5/6)*y*y)+((2/5)*x*y)
    #z = 3 * x ** 2- 6 * x * y + 4 * y ** 2 + 12 * x - 18 * y + 21
    #z = (1 - x)**2 + 100 * (y - x**2)**2
    #z = 7 * x* y/exp(x**2 + y**2)
    return z

def dfdx(x, y):
    return float(2/3) * x + float(1/5) * y
    #return 6*x-6*y+12
    #return	2 * (x - 1) - 4 * 100 * x * (y - x**2)
    #return -14*x**2 * y * exp(-x**2-y**2) + 7 * y* exp(-x**2-y**2)


def dfdy(x, y):
    return float(2/3) * y + float(1/5) * x

def gradf(x, y):
    return array([dfdx(x, y), dfdy(x, y)])


def grad_descent2_const(f, gradf, init_t, alpha):
    EPS = 1e-7
    prev_t = init_t-10*EPS
    t = init_t.copy()
    print("grad_descent2_const " + str(0.01))
    curve_x =[t]
    max_iter = 1000000
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        tCopy = t.copy()
        curve_x.append(tCopy)
        prev_t = t.copy()
        t -= 0.01*gradf(t[0], t[1])
        iter += 1
    print("iterations grad_descent2_const " + str(iter))
    print("min " + str(t))
    return np.array(curve_x)

def grad_descent2_split(f, gradf, init_t, alpha):
    EPS = 1e-7
    prev_t = init_t-10*EPS
    t = init_t.copy()
    print("grad_descent2_split " + str(0.01))
    curve_x =[t]
    max_iter = 1000000
    iter = 0
    g1 = dfdx(t[0],t[1])
    g2 = dfdy(t[0],t[1])
    while norm(t - prev_t) > EPS and iter < max_iter:
        y1= t[0] - 0.01*g1
        y2= t[1] - 0.01*g2
        f1 = f(y1,y2)
        f2 =f(t[0],t[1])
        if f1 < f2:
            tCopy = t.copy()
            prev_t = t.copy()
            curve_x.append(tCopy)
            t[0] = y1
            t[1] = y2
            g1 = dfdx(t[0],t[1])
            g2 = dfdy(t[0],t[1])
        else:
            alpha = float(0.01)
            #print(alpha)
        iter += 1
    print("iterations grad_descent2_split " + str(iter))
    print("min " + str(t))
    return np.array(curve_x)

def create_mesh(f):
    x = np.arange(-2, 2, 0.1)
    y = np.arange(-2,2, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i in mesh_size:
        for j in mesh_size:
            x_coor = X[i][j]
            y_coor = Y[i][j]
            Z[i][j] = f(x_coor, y_coor)
    return X, Y, Z

def plot_contour(ax, X, Y, Z):
    ax.set(
        title='Path',
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax.axis('square')
    return ax

init_point=np.array([15.0,-15.0])
xs=grad_descent2_split(f, gradf,init_point,2)
xs.reshape(-1,1)
fig, ax = plt.subplots(figsize=(6, 6))
X, Y, Z = create_mesh(f)
ax = plot_contour(ax, X, Y, Z)
ax.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
ax.plot(xs[-1,0], xs[-1,1], 'ro')
plt.show()