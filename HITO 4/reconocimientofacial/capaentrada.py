import cv2 as cv
import os
import imutils
modelo='FotosPrueba' #Nombre de la carpeta
ruta1='C:/Users/PC/Desktop/reconocimientofacial/Data' #Ruta donde se guardaran las fotos
rutacompleta = ruta1 + '/'+ modelo #Ruta completa
if not os.path.exists(rutacompleta): #Si no existe la ruta la crea
    os.makedirs(rutacompleta) #Crea la ruta
camara=cv.VideoCapture('orochi.mp4') #Captura video
ruidos=cv.CascadeClassifier('C:/Users/PC/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml') #clasificador de Haar de opencv
id=0 #Id de la imagen
while True: #Ciclo infinito
    respuesta,captura=camara.read() #Lee la camara
    if respuesta==False:break #Si no hay respuesta se rompe el ciclo
    captura=imutils.resize(captura,width=640) #Redimensiona la captura

    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY) #Convierte a grises
    idcaptura=captura.copy() #Copia la captura

    cara=ruidos.detectMultiScale(grises,1.3,5) #Detecta la cara

    for(x,y,e1,e2) in cara: #Ciclo para recorrer la cara
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,255,0),2) #Dibuja un rectangulo
        rostrocapturado=idcaptura[y:y+e2,x:x+e1] #Rostro capturado
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC) #Redimensiona el rostro
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocapturado) #Guarda la imagen
        id=id+1 #Aumenta el id
    
    cv.imshow("Resultado rostro", captura) #Muestra el resultado

    if id==350: #Si el id es igual a 350 se rompe el ciclo
        break 
camara.release() #Libera la camara
cv.destroyAllWindows() #Cierra las ventanas