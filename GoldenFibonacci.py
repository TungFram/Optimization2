import numpy as np
import matplotlib.pyplot as plt


def gss(f, a, b, tol=1e-7):
    phi = (np.sqrt(5) + 1) / 2
    d = b - (b - a) / phi
    c = a + (b - a) / phi
    while abs(d - c) > tol:
        if f(d) < f(c):
            b = c
        else:
            a = d
        d = b - (b - a) / phi
        c = a + (b - a) / phi
    return (a + b) / 2


def f(X):
    x, y = X
    # z = float(1/3) * x**2 + float(1/3) * y**2 + float(1/5) * y *x
    # z = 3*x*x-6*x*y+4*y*y+12*x-18*y+21
    # z = (1 - x)**2 + 100 * (y - x**2)**2
    z = (5*x) / (x*x + y * y + 2)
    return z


def dfdx(x, y):
    # return float(2/3) * x + float(1/5) * y
    # return 6*x-6*y+12
    # return	2 * (x - 1) - 4 * 100 * x * (y - x**2)
    return -10*((x*x)/((x*x+y*y+2)*(x*x+y*y+2)))+(5/(x*x+y*y+2))


def dfdy(x, y):
    # return float(2/3) * y + float(1/5) * x
    # return -6*x+8*y-18
    # return	2 * 100 * (y - x**2)
    return -10*x*((y)/((x*x+y*y+2)*(x*x+y*y+2)))


def gradf(X):
    x, y = X
    return np.array([dfdx(x, y), dfdy(x, y)])


def create_mesh(f):
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i in mesh_size:
        for j in mesh_size:
            x_coor = X[i][j]
            y_coor = Y[i][j]
            Z[i][j] = f(np.array([x_coor, y_coor]))
    return X, Y, Z


def plot_contour(ax, X, Y, Z, alpha):
    ax.set(
        title='Trajectory with rate ' + str(alpha),
        xlabel='x1',
        ylabel='x2'
    )

    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax.axis('square')
    return ax


def gradient_descent_gold(J, J_grad, x_init, epsilon=1e-7, max_iterations=1000):
    print("a " + str(a))
    x = x_init
    curve_x = [x_init]
    num_iter = 0
    for i in range(max_iterations):
        q = lambda alpha: J(x - alpha * J_grad(x))
        alpha = gss(q, 0.0, a)
        x = x - alpha * J_grad(x)
        curve_x.append(x)
        if np.linalg.norm(J_grad(x)) < epsilon:
            print("min gradient_descent_gold " + str(curve_x[-1]))
            print("iterations gradient_descent_gold success " + str(num_iter))
            return np.array(curve_x)
        num_iter+=1
    print("iterations gradient_descent_gold " + str(num_iter))
    print("min gradient_descent_gold " + str(curve_x[-1]))
    return np.array(curve_x)


def gradient_descent_fib(J, J_grad, x_init, epsilon=1e-7, max_iterations=1000):
    print("a " + str(a))
    x = x_init
    curve_x = [x_init]
    num_iter = 0
    for i in range(max_iterations):
        q = lambda alpha: J(x - alpha * J_grad(x))
        alpha = fib(q, 0.0, a)
        x = x - alpha * J_grad(x)
        curve_x.append(x)
        if np.linalg.norm(J_grad(x)) < epsilon:
            print("min gradient_descent_fib " + str(curve_x[-1]))
            print("iterations gradient_descent_fib success " + str(num_iter))
            return np.array(curve_x)
        num_iter += 1
    print("iterations gradient_descent_fib " + str(num_iter))
    print("min gradient_descent_fib " + str(curve_x[-1]))
    return np.array(curve_x)


def initSequence(num):

    fibonacci =[]
    fibonacci.append(0)
    fibonacci.append(1)
    k = 2
    while (fibonacci[-1] <= num):
        a = fibonacci[k - 1] + fibonacci[k - 2]
        fibonacci.append(a)
        k += 1
    return fibonacci


def fib(f, start, end, eps=1e-7):
    min = 0
    eps *= 2
    array = initSequence((end - start) / eps)
    n = len(array) -1 ;
    n1 = n
    c = start + float(array[n - 2] / array[n]) * (end - start)
    d = start + float(array[n -1] / array[n]) * (end - start)
    fc = f(c)
    fd = f(d)
    for k in range(n1):
        n -= 1
        if n == 1:
            n1 = k
            break

        if fc < fd:
            end = d
            d = c
            fd = fc
            c = start + array[n - 2] / array[n] * (end - start)
            fc = f(c)

        else:
            start = c
            c = d
            fc = fd
            d = start + array[n - 1] / array[n] * (end - start)
            fd = f(d)
    return (start + end) / 2

a = 1
x_init = np.array([-4.0, -1.0])
xs = gradient_descent_fib(f, gradf, x_init, max_iterations=40000)
# xs = gradient_descent_gold(f, gradf, x_init, max_iterations=40000)

fig, ax = plt.subplots(figsize=(6, 6))
X, Y, Z = create_mesh(f)
ax = plot_contour(ax, X, Y, Z, a)
xs = np.array(xs)
ax.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
ax.plot(xs[-1,0], xs[-1,1], 'ro')
plt.show()