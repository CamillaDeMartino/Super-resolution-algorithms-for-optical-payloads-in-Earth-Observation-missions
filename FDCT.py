import cv2
import numpy as np
from curvelops import FDCT2D
from curvelops import fdct2d_wrapper as ct
from curvelops.plot import curveshow
from matplotlib import pyplot as plt


# Carica l'immagine in scala di grigi
logo = cv2.imread("area_centrale.png", cv2.IMREAD_GRAYSCALE)

logo_gray = logo.swapaxes(0, 1)
# Visualizza l'immagine utilizzando Matplotlib
cv2.imshow("Imag", logo_gray.swapaxes(0, 1))
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Img dims: ", logo_gray.shape)

# Normalizza i valori dei pixel dell'immagine tra 0 e 1
logo_gray = logo_gray.astype(float) / 255.0

# Applica la trasformata di curvelet 2D all'immagine
C2D = FDCT2D(logo_gray.shape, nbscales=4, nbangles_coarse=4, allcurvelets=False)

# Estrai i coefficienti di curvelet
logo_c = C2D.struct(C2D @ logo_gray)
lunghezza_lista = len(logo_c)
print("La lista logo_c contiene", lunghezza_lista, "elementi.")
print("Dims prima matrice: ", logo_c[0][0].shape)
print("Dims seconda matrice: ", logo_c[1][0].shape)
print("Dims terza matrice: ", logo_c[2][0].shape)
print("Dims quarta matrice: ", logo_c[3][0].shape)


print("Angles: ", len(logo_c[3][0]))

# Visualizza le curvelet

fig_axes = curveshow(logo_c, kwargs_imshow=dict(extent=[0, 1, 1, 0]))
#plt.show()


#--------------------INVERSA------------------

x = logo
logo = logo.astype(float) / 255.0
# Genera numeri casuali distribuiti secondo una distribuzione normale
random_array = np.random.normal(0.0, 1.0, logo.shape)

# Moltiplica l'array di numeri casuali per l'immagine normalizzata
result = random_array * logo
c = ct.fdct2d_forward_wrap(4, 8, False, result)
fig_axes = curveshow(c, kwargs_imshow=dict(extent=[0, 1, 1, 0]))
#plt.show()
lunghezza_lista = len(c)

print("La lista c contiene", lunghezza_lista, "elementi.")
print("Dims prima matrice: ", c[0][0].shape)
print("Dims seconda matrice: ", c[1][0].shape)
print("Dims terza matrice: ", c[2][0].shape)
print("Dims quarta matrice: ", c[3][0].shape)

xinv = ct.fdct2d_inverse_wrap(*x.shape, 4, 8, False, c[3][0])

cv2.imshow("Inverse Image", xinv )
cv2.waitKey(0)
cv2.destroyAllWindows()






    
