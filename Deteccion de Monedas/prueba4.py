import cv2
import numpy as np
from scipy import ndimage
from scipy.integrate import odeint

# Función de difusión para suavizar la imagen
def diff_eqn(u, t, D):
    u = u.reshape(gris.shape)
    laplacian = cv2.Laplacian(u, cv2.CV_64F)
    return (D * laplacian).flatten()

# Parámetros del problema
D = 0.5  # Coeficiente de difusión
umbral = 0.5  # Umbral para la detección de bordes

# Cargar imagen y convertirla a escala de grises
img = cv2.imread('monedas.jpg')
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resolver la ecuación de difusión para suavizar la imagen
t = np.linspace(0, 1, 10)
u = ndimage.grey_erosion(gris, size=(3, 3))  # Imagen inicial para la ecuación de difusión
u = odeint(diff_eqn, u.flatten(), t, args=(D,)).reshape(-1, *gris.shape)

# Convertir la imagen resultante de la ecuación de difusión a 8 bits
u_8bit = np.uint8(u[-1])

# Detectar bordes mediante Canny
umbral1 = int(umbral * np.max(u_8bit))
umbral2 = int(1.2 * umbral1)
bordes = cv2.Canny(u_8bit, umbral1, umbral2)

# Aplicar operaciones morfológicas para mejorar la detección de bordes
kernel = np.ones((3, 3), np.uint8)
bordes = cv2.dilate(bordes, kernel, iterations=2)
bordes = cv2.erode(bordes, kernel, iterations=2)

# Encontrar contornos y dibujarlos en la imagen original
contornos, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contornos, -1, (0, 0, 255), 2)

# Mostrar la imagen resultante
cv2.imshow('Monedas detectadas', img)
cv2.waitKey(0)
