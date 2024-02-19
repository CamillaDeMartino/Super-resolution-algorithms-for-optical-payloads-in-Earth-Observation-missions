import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from curvelops import FDCT2D
from curvelops.plot import curveshow


#A - Campionamento Quincunx 
def quincunx_sampling(image):

    row, column = image.shape

    # Eseguire uno spostamento di mezzo pixel in x e y
    I1 = image[0:row:2, 0:column:2]
    I2 = image[1:row:2, 1:column:2]

    return I1, I2

#B - Combinare frame I1 e I2
def combine_frames(I1, I2):
    
    # Calcola le coordinate xh, yh per la griglia ad alta risoluzione
    xh, yh = np.meshgrid(np.arange(I1.shape[1]), np.arange(I1.shape[0]))

    # Calcola le coordinate x1, y1, x2, y2 come specificato nell'equazione
    x1 = 2 * xh + 1
    y1 = 2 * yh + 1
    x2 = 2 * xh
    y2 = 2 * yh

    # Calcola le dimensioni della griglia Quincunx
    rows, cols = I1.shape

    # Inizializza l'immagine risultante
    H = np.zeros((2 * rows, 2 * cols), dtype=np.uint8)

    # Posizionare i pixel a bassa risoluzione sulla griglia ad alta risoluzione
    H[y1, x1] = I1
    H[y2, x2] = I2

    #Trasformazione = invertire le matrici???
    result_image = H  

    return result_image




#C - Rotate image by 45 degree
def rotate_quincunx_image(image):

    # prendi le dimensioni dell'immagine e determina il centro
    rows, columns = image.shape
    cX, cY = (columns // 2, rows // 2)

    # prendi la matrice di rotazione (applicando il negativo dell'angolo per ruotare in senso orario), quindi prendi il seno e il coseno
    # (ovvero, le componenti di rotazione della matrice)
    M = cv2.getRotationMatrix2D((cX, cY), -45, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # calcola le nuove dimensioni dell'immagine
    nR = int((rows * sin) + (columns * cos))
    nC = int((rows * cos) + (columns * sin))

    # regolare la matrice di rotazione per tenere conto della traslazione
    M[0, 2] += (nR / 2) - cX
    M[1, 2] += (nC / 2) - cY

    # esegui la rotazione effettiva e restituire l'immagine
    rotated = cv2.warpAffine(image, M, (nR, nC))

    #rotated = imutils.rotate_bound(image, 45)

    return rotated


#D - Up-Sampling
def up_sampling(image):
   
    # Fattore Up-sampling 
    f = 2

    m,n = image.shape
    m_new = m*f
    n_new = n*f

    # Crea matrice di zeri per la matrice di sovracampionamento 
    img_upsampled = np.zeros((m_new, n_new), dtype=np.uint8) 
    for i in range(m): 
        for j in range(n): 
            img_upsampled[i*f, j*f] = image[i, j]


    #img_upsampled = cv2.resize(image, (m_new, n_new))

    return img_upsampled





#--------------------MAIN-----------------------


# Carica un'immagine ad alta risoluzione
image_hr = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Image Hr", image_hr)
cv2.waitKey(0)
print("Size original: ", image_hr.shape)
print("Pixel (0,0): ", image_hr[0, 0])
print("Pixel (0,1): ", image_hr[0, 1])
print("Pixel (0,2): ", image_hr[0, 2])
print("Pixel (1,1): ", image_hr[1, 1])
print("Pixel (1,3): ", image_hr[1, 3])

# Applicare il campionamento Quincunx
I1, I2 = quincunx_sampling(image_hr)

# Visualizzare le immagini risultanti
cv2.imshow("I1 - Quincunx Sampling", I1)
print("\nI1 size: ", I1.shape)
print("I1 Pixel (0,0): ", I1[0, 0])
print("I1 Pixel (0,1): ", I1[0, 1])

cv2.imshow("I2 - Quincunx Sampling", I2)
print("\nI2 size: ", I2.shape)
print("I2 Pixel (0,0): ", I2[0, 0])
print("I2 Pixel (0,1): ", I2[0, 1])

cv2.waitKey(0)


#Combina i due frame LR
HR = combine_frames(I1, I2)

cv2.imshow("HR Sampling", HR)
print("\nHR size: ", HR.shape)
print("HR Pixel (0,0): ", HR[0, 0])
print("HR Pixel (0,1): ", HR[0, 1])
print("HR Pixel (0,2): ", HR[0, 2])
print("HR Pixel (1,1): ", HR[1, 1])
print("HR Pixel (1,3): ", HR[1, 3])
cv2.waitKey(0)

cv2.destroyAllWindows()


HR_rotate = rotate_quincunx_image(HR)
#cv2.imshow("HR Rotated", HR_rotate)
#cv2.waitKey(0)

plt.figure(figsize=(12, 8))
plt.imshow(HR_rotate, cmap='gray')
plt.title('HR Rotated')
plt.show()

HR_upsampling = up_sampling(HR_rotate)
print("\nSize ups: ", HR_upsampling.shape)
#cv2.imshow("HR Up-Sampling", HR_upsampling)
#cv2.waitKey(0)

plt.figure(figsize=(12, 8))
plt.imshow(HR_upsampling, cmap='gray')
plt.title('HR Up-Sampling')
plt.show()



# Calcola la trasformata curvelet discreta
nbscales = 6
nbangles_coarse = 16
# Definisci la trasformata delle curvelet (FDCT) in base alle dimensioni dell'immagine HR_upsampling
FDCT = FDCT2D(dims=HR_upsampling.shape, nbscales=6, nbangles_coarse=16, allcurvelets=True)

# Applica la trasformata delle curvelet all'immagine HR_upsampling
# restituendo i coefficienti della trasformata.
c = FDCT @ HR_upsampling

print("Coefff", c.shape)

# Calcola l'inversa della trasformata delle curvelet per ricostruire l'immagine originale
FDCT_inv = FDCT.H @ c

# Verifica che l'immagine originale e quella ricostruita siano simili entro una certa tolleranza
#np.testing.assert_allclose(HR_upsampling, FDCT_inv, rtol=1e-6, atol=1e-8)
#np.testing.assert_allclose(HR_upsampling, FDCT_inv)

# Calcola l'errore quadratico medio (RMSE)
rmse = np.sqrt(np.mean((HR_upsampling - FDCT_inv)**2))

# Stampa il RMSE
print("RMSE:", rmse)

