import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import product
from numpy.linalg import norm
def object_function(xk):
    x,y = xk
    #f = 100 * (xk[0] ** 2 - xk[1]) ** 2 + (xk[0] - 1) ** 2
    f = 2*x*x+((5/6)*y*y)+((2/5)*x*y)
    #f =3*x*x-6*x*y+4*y*y+12*x-18*y+21
    return f


def gradient_function(xk):
    x,y = xk
    # gk = np.array([
    #	400 * (xk[0] ** 2 - xk[1]) * xk[0] + 2 * (xk[0] - 1), #	-200 * (xk[0] ** 2 - xk[1])
    # ])
    gk = np.array([
        float(2/3) * x + float(1/5) * y,
        float(2/3) * y + float(1/5) * x
    ])
    # gk = np.array([ #	6*x-6*y+12,
    #	-6*x+8*y-18 # ])
    return gk

def wolfe_powell(xk, sk):
    alpha = 1.0
    a = 0.0
    b = -sys.maxsize
    c_1 = 0.1
    c_2 = 0.5
    k = 0
    while k < 100:
        k += 1
        if object_function(xk) - object_function(xk + alpha * sk) >= -c_1 * alpha * np.dot(gradient_function(xk), sk):
            # print ('Выполнить условие 1')
            if np.dot(gradient_function(xk + alpha * sk), sk) >= c_2 * np.dot(gradient_function(xk), sk):
                # print ('Условие 2 выполнено')
                return alpha
            else:
                a = alpha
                alpha = min(2 * alpha, (alpha + b) / 2)
        else:
            b = alpha
            alpha = 0.5 * (alpha + a)
    return alpha



def create_mesh(f):
    x = np.arange(-5, 5, 0.025)
    y = np.arange(-5, 5, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i in mesh_size:
        for j in mesh_size:
            x_coor = X[i][j]
            y_coor = Y[i][j]
            Z[i][j] = f(np.array([x_coor, y_coor]))
    return X, Y, Z

def plot_contour(ax, X, Y, Z):
    ax.set(
        title='Conjugative Gradient Method',
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax.axis('square')
    return ax

# Метод сопряженного градиента
def conjugate_gradient(x0, eps):
    xk = x0
    gk = gradient_function(xk)
    sigma = np.linalg.norm(gk)
    sk = -gk
    step = 0
    prev_xk = xk-10 * eps
    w = np.zeros ((2, 10 ** 3)) # Сохраняем итерацию и устанавливаем переменную xk
    curve_x = [x0]
    while norm(xk - prev_xk) > eps and step < 100:
        prev_xk = xk.copy()
        w[:, step] = np.transpose(xk)
        step += 1
        alpha = wolfe_powell(xk, sk)
        xk = xk + alpha * sk
        curve_x.append(xk)
        g0 = gk
        gk = gradient_function(xk)
        miu = (np.linalg.norm(gk) / np.linalg.norm(g0))**2
        sk = -1 * gk + miu * sk
        sigma = np.linalg.norm(gk)
    print("conjugate gradient iterations" + str(step))
    print("conjugate gradient min" + str(curve_x[-1]))
    return np.array(curve_x)

eps = 1e-7
x0 = np.array([-3.0,-4.0])
xs = conjugate_gradient(x0, eps)

fig, ax = plt.subplots(figsize=(6, 6))
X, Y, Z = create_mesh(object_function)
ax = plot_contour(ax, X, Y, Z)
ax.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
ax.plot(xs[-1,0], xs[-1,1], 'ro')
plt.show()