import cv2
import numpy as np
from scipy.integrate import solve_ivp

# Resolver una EDO simple
def my_edo(t, y):
    return t * y

interval = (0, 2)
initial_condition = [1]
sol = solve_ivp(my_edo, interval, initial_condition, t_eval=np.linspace(interval[0], interval[1], 100))

# Utilizar una parte de la solución de la EDO para ajustar el valor de umbral
threshold_scale = sol.y[0][50]

valorGauss = 1
valorKernel = 7
original = cv2.imread('monedas.jpg')
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gris, (valorGauss, valorGauss), 0)

# Ajustar el valor de umbral utilizando la solución de la EDO
umbral_1 = int(60 * threshold_scale)
umbral_2 = int(100 * threshold_scale)
canny = cv2.Canny(gauss, umbral_1, umbral_2)

kernel = np.ones((valorKernel, valorKernel), np.uint8)
cierre = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

contornos, jerarquía = cv2.findContours(cierre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("monedas encontradas: {}".format(len(contornos)))
cv2.drawContours(original, contornos, -1, (0, 0, 255), 2)

cv2.imshow("Resultado", original)
cv2.waitKey(0)
