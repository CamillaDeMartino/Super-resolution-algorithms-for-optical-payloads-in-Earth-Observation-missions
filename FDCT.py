import cv2
import numpy as np
from curvelops import FDCT2D
from curvelops import fdct2d_wrapper as ct
from matplotlib import pyplot as plt


# Carica l'immagine in scala di grigi
logo = cv2.imread("area_centrale.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imag", logo)
cv2.waitKey(0)
cv2.destroyAllWindows()

   

c = ct.fdct2d_forward_wrap(4, 8, True, logo)
xinv = ct.fdct2d_inverse_wrap( *logo.shape, 4, 8, True, c)

plt.figure(figsize=(12, 8))
plt.imshow(np.real(xinv), cmap='gray')
plt.title('inv')
plt.show()




    
