import cv2 as cv
import os
import imutils
dataRuta='C:/Users/PC/Desktop/reconocimientofacial/Data' #Ruta de la carpeta Data
listaData=os.listdir(dataRuta) #Lista de las carpetas dentro de la carpeta Data
entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create() #Creando el reconocedor
entrenamientoEigenFaceRecognizer.read('EntrenamientoEigenFaceRecognizer.xml') #Leyendo el entrenamiento
ruidos=cv.CascadeClassifier('C:/Users/PC/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml') #clasificador de Haar de opencv
camara=cv.VideoCapture('orochi.mp4') #Iniciando la camara
while True: 
    respuesta,captura=camara.read() #Capturando la imagen
    if respuesta==False:break #Si no hay imagen se rompe el ciclo
    captura=imutils.resize(captura,width=640) #Redimensiona la captura
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY) #Convierte a grises
    idcaptura=grises.copy() #Copia la captura
    cara=ruidos.detectMultiScale(grises,1.3,5) #Detecta la cara
    for(x,y,e1,e2) in cara: #Ciclo para recorrer la cara
        rostrocapturado=idcaptura[y:y+e2,x:x+e1] #Rostro capturado
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC) #Redimensiona el rostro
        resultado=entrenamientoEigenFaceRecognizer.predict(rostrocapturado) #Predice el resultado
        cv.putText(captura, '{}'.format(resultado), (x,y-5), 1,1.3,(0,255,0),1,cv.LINE_AA) #Pone el texto
        if resultado[1]<8000: #Si el resultado es menor a 8000 es una cara conocida
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x,y-20), 2,1.1,(0,255,0),1,cv.LINE_AA) #Pone el texto
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2) #Dibuja un rectangulo
        else:
            cv.putText(captura,"No encontrado", (x,y-20), 2,0.7,(0,255,0),1,cv.LINE_AA) #Pone el texto
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2) #Dibuja un rectangulo
    cv.imshow("Resultados", captura) #Muestra el resultado
    if cv.waitKey(1)==ord('s'): #Si se presiona la tecla s se rompe el ciclo
        break
camara.release()
cv.destroyAllWindows()


