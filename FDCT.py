import cv2
import numpy as np
from curvelops import FDCT2D
from curvelops import fdct2d_wrapper as ct
from matplotlib import pyplot as plt
from pylops.utils import dottest
from curvelops.utils import (
    apply_along_wedges,
    array_split_nd,
    energy,
    energy_split,
    ndargmax,
    split_nd,
)

# Add 1 to each wedge
def modify_wedge(c_wedge, w_index, s_index, num_wedges_scale, num_scales):
    # Implementa qui la logica desiderata per modificare i valori dei cunei
    print("Prova2: ", (w_index))
    modified_wedge = c_wedge * (w_index + 1)  # Ad esempio, moltiplica ogni cuneo per il suo indice nella scala più 1
    return modified_wedge
    
# Carica l'immagine in scala di grigi
x = cv2.imread("/home/camilla/Scrivania/Tesi/Images/test_image.png", cv2.IMREAD_GRAYSCALE)
shape = x.shape
Cop = FDCT2D(shape, nbscales=4, nbangles_coarse=4,allcurvelets=False)
# Create a vector of curvelet coeffs
y = Cop @ x
# Convert to structure
y_struct = Cop.struct(Cop @ x)
print(len(y_struct))


# Utilizza la nuova funzione per modificare i wedges
y_struct_one = apply_along_wedges(
    y_struct[-1],
    modify_wedge
)

print(len(y_struct_one))
# Convert back to vector
y_one = Cop.vect(y_struct_one)

# Ensure that each wedge of the modified wedge - original is
# equal to 2d array of ones

xinv = Cop.H * y_one

plt.figure(figsize=(12, 8))
plt.imshow(np.abs(xinv), cmap='gray')
plt.title('inv')
plt.show()
