import cv2
import numpy as np
from curvelops import FDCT2D
from curvelops import fdct2d_wrapper as ct
from matplotlib import pyplot as plt


# Carica l'immagine in scala di grigi
logo = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)


print("Prima: ", logo[0, 0])
cv2.imshow("Imag", logo)
cv2.waitKey(0)
cv2.destroyAllWindows()

   

c = ct.fdct2d_forward_wrap(4, 8, False, logo)
print("Poco prima: ", np.abs(c[3][0][0, 0]))
xinv = ct.fdct2d_inverse_wrap( *logo.shape, 4, 8, False, c)
print("Dopo: ", np.abs(xinv[0,0]))

plt.figure(figsize=(12, 8))
plt.imshow(np.abs(xinv), cmap='gray')
plt.title('inv')
plt.show()




    
