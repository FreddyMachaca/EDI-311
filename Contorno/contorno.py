import cv2
imagen=cv2.imread('contorno.jpg')
grises=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY) #Convertir a grises
_,umbral=cv2.threshold(grises,100,255,cv2.THRESH_BINARY) #Umbral
contorno,jerarquía = cv2.findContours(umbral,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #Contorno
cv2.drawContours(imagen,contorno,-1,(251, 63, 52),3) #Dibujar contorno
#Mostrar
cv2.imshow('Imagen original',imagen) 
#cv2.imshow('Imagen en grises',grises)
#cv2.imshow('Imagen Umbral',umbral)
cv2.waitKey(0)
cv2.destroyAllWindows()