import cv2
import numpy as np

# Cargar los archivos del modelo preentrenado
modelo = 'mobilenet_iter_73000.caffemodel'
configuracion = 'deploy.prototxt'

# Inicializar la red neuronal con el modelo SSD y MobileNet
red = cv2.dnn.readNetFromCaffe(configuracion, modelo)

# Iniciar la captura de video
#2 es para iphone
#1 es para mac
cap = cv2.VideoCapture(0)  # '0' es generalmente la cámara web predeterminada

while True:
    # Leer un cuadro del video
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pasar el blob a la red y obtener las detecciones y predicciones
    red.setInput(blob)
    detecciones = red.forward()

    contador_personas = 0  # Inicializar el contador de personas

    # Bucle sobre las detecciones
    for i in np.arange(0, detecciones.shape[2]):
        confianza = detecciones[0, 0, i, 2]

        if confianza > 0.2:  # Umbral de confianza
            idx = int(detecciones[0, 0, i, 1])
            if idx == 15:  # Clase 15 corresponde a 'persona'
                contador_personas += 1  # Incrementar el contador de personas

                # Calcular las coordenadas (x, y) del cuadro delimitador para el objeto
                caja = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
                (inicioX, inicioY, finX, finY) = caja.astype("int")

                # Dibujar el cuadro delimitador alrededor de la persona detectada
                cv2.rectangle(frame, (inicioX, inicioY), (finX, finY), (0, 255, 0), 2)

    # Mostrar el conteo de personas
    texto = f"Personas detectadas: {contador_personas}"
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar el cuadro con las detecciones
    cv2.imshow("Detección en Vivo", frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
