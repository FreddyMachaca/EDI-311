import cv2 #cv2 es utilizado para la manipulación de imágenes
import math #math es utilizado para realizar operaciones matemáticas

def object_size(x): # Función para calcular el tamaño del objeto en función de la posición horizontal
    # Coeficientes de la ecuación
    p = 2/x #representa 
    q = 3/x
    
    # Factor integrante
    factor = x**2
    
    # Solución de la ecuación
    C = 1  # Constante de integración
    y = (3/2)*x + C/x**2
    
    return y

cap = cv2.VideoCapture(0)  # Iniciar la webcam

while True:
    ret, frame = cap.read()  # Leer un fotograma de la webcam
    if not ret:
        break
    
    height, width, channels = frame.shape  # Obtener las dimensiones del fotograma
    center_x = width/2  # Calcular la posición del centro horizontal
    
    # Calcular el tamaño del objeto en función de la posición horizontal
    size = object_size(center_x)
    
    # Dibujar un rectángulo en el centro de la imagen con el tamaño calculado
    rect_width = int(size * width)  # Ancho del rectángulo
    rect_height = int(size * height)  # Altura del rectángulo
    rect_x = int(center_x - rect_width/2)  # Posición horizontal del rectángulo
    rect_y = int(height/2 - rect_height/2)  # Posición vertical del rectángulo
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x+rect_width, rect_y+rect_height), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)  # Mostrar el fotograma con el rectángulo
    #mostrar en pantalla los valores de las variables
    print("width: ", width)
    print("height: ", height)
    print("center_x: ", center_x)
    print("size: ", size)
    print("rect_width: ", rect_width)
    print("rect_height: ", rect_height)
    print("rect_x: ", rect_x)
    print("rect_y: ", rect_y)
    
    if cv2.waitKey(1) == ord('q'):  # Salir con 'q'
        break

cap.release()  # Liberar la webcam
cv2.destroyAllWindows()  # Cerrar las ventanas
