import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_edo(fun, interval, initial_condition, num_points=100):
    t_eval = np.linspace(interval[0], interval[1], num_points)
    sol = solve_ivp(fun, interval, initial_condition, t_eval=t_eval)
    return sol

def plot_solution(solution, label):
    plt.plot(solution.t, solution.y[0], label=label)

# Ejemplo de uso
# Definir la función de la EDO y los parámetros necesarios
def my_edo(t, y):
    return t * y

interval = (0, 2)
initial_condition = [1]

# Resolver y graficar la solución
sol = solve_edo(my_edo, interval, initial_condition)
plt.figure()
plot_solution(sol, 'Mi EDO')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
