import cv2
import matplotlib.pyplot as plt

# Cargar imágenes
# Asegúrate de reemplazar 'ruta_a_tu_imagen1.jpg' y 'ruta_a_tu_imagen2.jpg' con las rutas correctas de tus imágenes
img1 = cv2.imread('ruta_a_tu_imagen1.jpg')
img2 = cv2.imread('ruta_a_tu_imagen2.jpg')

# Convertir imágenes a escala de grises
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Inicializar SIFT
sift = cv2.SIFT_create()

# Encontrar los puntos clave y descriptores con SIFT
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Crear emparejador de características
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Emparejar descriptores
matches = bf.match(descriptors1, descriptors2)

# Ordenarlos según su distancia
matches = sorted(matches, key=lambda x: x.distance)

# Dibujar los primeros 10 emparejamientos
img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar la imagen
plt.imshow(img3)
plt.show()
