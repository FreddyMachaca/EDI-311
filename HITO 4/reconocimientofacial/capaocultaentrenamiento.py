import cv2 as cv
import os
import numpy as np
from time import time
dataRuta='C:/Users/PC/Desktop/reconocimientofacial/Data' #Ruta de la carpeta Data
listaData=os.listdir(dataRuta) #Lista de las carpetas dentro de la carpeta Data
#print('data',listaData) 
ids=[] #Lista de identificadores
rostrosData=[] #Lista de rostros
id=0 #Identificador
tiempoInicial=time() #Tiempo inicial
for fila in listaData: #Recorriendo las carpetas dentro de la carpeta Data
    rutacompleta=dataRuta+'/'+ fila #Ruta completa de la carpeta
    print('Iniciando lectura...') #Mensaje
    for archivo in os.listdir(rutacompleta): #Recorriendo las imagenes dentro de la carpeta
        
        print('Imagenes: ',fila +'/'+archivo) #Mensaje
    
        ids.append(id) #Agregando el id a la lista de identificadores
        rostrosData.append(cv.imread(rutacompleta+'/'+archivo,0)) #Agregando el rostro a la lista de rostros
    id=id+1 #Aumentando el id
    tiempofinalLectura=time() #Tiempo final de lectura
    tiempoTotalLectura=tiempofinalLectura-tiempoInicial #Tiempo total de lectura
    print('Tiempo total lectura: ',tiempoTotalLectura) #Mensaje

entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create() #Creando el reconocedor
print('Iniciando el entrenamiento...espere') #Mensaje
entrenamientoEigenFaceRecognizer.train(rostrosData,np.array(ids)) #Entrenando el reconocedor
TiempofinalEntrenamiento=time() #Tiempo final de entrenamiento
tiempoTotalEntrenamiento=TiempofinalEntrenamiento-tiempoTotalLectura #Tiempo total de entrenamiento
print('Tiempo entrenamiento total: ',tiempoTotalEntrenamiento) #Mensaje
entrenamientoEigenFaceRecognizer.write('EntrenamientoEigenFaceRecognizer.xml') #Guardando el entrenamiento
print('Entrenamiento concluido') #Mensaje