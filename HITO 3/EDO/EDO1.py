import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Ejercicio 1: Ecuación diferencial ordinaria separable de primer orden
def fun1(t, y):
    return (2 * t + 3) * (y**2 - 4)

sol1 = solve_ivp(fun1, (0, 1), [-1], t_eval=np.linspace(0, 1, 100))

# Ejercicio 2: Ecuación diferencial ordinaria exacta de primer orden
# Reescribimos la ecuación como y'(x) = -M(x, y) / N(x, y)
def fun2(t, y):
    M = 2 * y**2 - 3
    N = 2 * t * y**2 + 4
    return -M / N

sol2 = solve_ivp(fun2, (0, 1), [1], t_eval=np.linspace(0, 1, 100))

# Ejercicio 3: Ecuación diferencial ordinaria separable de primer orden
# Reescribimos la ecuación como y'(x) = (x/y^2)
def fun3(t, y):
    return t / y**2

sol3 = solve_ivp(fun3, (0, 1), [1], t_eval=np.linspace(0, 1, 100))

# Gráficas de las soluciones
plt.figure()
plt.plot(sol1.t, sol1.y[0], label='Ejercicio 1')
plt.plot(sol2.t, sol2.y[0], label='Ejercicio 2')
plt.plot(sol3.t, sol3.y[0], label='Ejercicio 3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
