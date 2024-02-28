import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from curvelops import FDCT2D
from curvelops.plot import curveshow
from itertools import combinations, cycle


#A - Campionamento Quincunx 
def quincunx_sampling(image):

    """row, column = image.shape

    # Eseguire uno spostamento di mezzo pixel in x e y
    I1 = image[0:row:2, 0:column:2]
    I2 = image[1:row:2, 1:column:2]"""

    I1 = cv2.imread("LR_1.png", cv2.IMREAD_GRAYSCALE)
    I2 = cv2.imread("LR_2.png", cv2.IMREAD_GRAYSCALE)

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
    H[y1, x1] = I2
    H[y2, x2] = I1

    #Trasformazione = invertire le matrici???
    result_image = H  

    return result_image




#C - Rotate image by 45 degree
def rotate_quincunx_image(image):

    rows, columns = image.shape
    new=np.zeros((rows*2,columns*2))
    
    for j in range(columns):
        for i in range(rows):
            new[i+j,columns-j+i]=image[j,i]
    

    return new


#D - Up-Sampling
def up_sampling(image_rotated, HR):
    
    f = 2

    m,n = HR.shape
    m_new = m*f
    n_new = n*f
    img_upsampled = cv2.resize(image_rotated, (m_new, n_new), interpolation=cv2.INTER_NEAREST)
    
    #img_upsampled = np.kron(image_rotated, np.ones((2, 2)))

    return img_upsampled



#E - Discrete Curvelet Domain 
def fdct(image):
    imag_swap = image.swapaxes(0, 1)

    print("Img dims: ", imag_swap.shape)

    # Normalizza i valori dei pixel dell'immagine tra 0 e 1
    imag_swap = imag_swap.astype(float) / 255.0

    # Applica la trasformata di curvelet 2D all'immagine
    C2D = FDCT2D(imag_swap.shape, nbscales=4, nbangles_coarse=4, allcurvelets=False)

    # Estrai i coefficienti di curvelet
    coeff = C2D.struct(C2D @ imag_swap)
    lunghezza_lista = len(coeff)
    print("\nLa lista coeff contiene", lunghezza_lista, "elementi.")

    print("Dims prima matrice: ", coeff[0][0].shape)
    print("Dims seconda matrice: ", coeff[1][0].shape)
    print("Dims terza matrice: ", coeff[2][0].shape)
    print("Dims quarta matrice: ", coeff[3][0].shape)

    print("Angles: ", len(coeff[3]))

    # Visualizza le curvelet
    fig_axes = curveshow(coeff, kwargs_imshow=dict(extent=[0, 1, 1, 0]))
    #plt.show()

    return coeff[3][0]



#F - Interpolation
def find_missing_pixels(image, zero):
    
    # Identifica le coordinate dei pixel mancanti
    missing_pixels = []
    rows, cols = image.shape[:2]
    for y in range(rows):
        for x in range(cols):
            # Verifica se il pixel è mancante (0)
            if np.abs(image[y, x]) < zero:
                missing_pixels.append((x, y))

             
    return missing_pixels


def divide_into_groups(selected_coefficients):

    # Genera tutte le possibili combinazioni di tre coefficienti
    coefficient_combinations = list(combinations(selected_coefficients, 3))
    
    # Dividi le combinazioni in gruppi
    groups = [[] for _ in range(4)]  # 4 gruppi vuoti
    
    # Ciclo attraverso tutte le combinazioni e assegnale ai gruppi
    combination_cycle = cycle(coefficient_combinations)
    for i in range(4):
        combination = next(combination_cycle)
        groups[i].append(combination)
    
    """# Stampa gli elementi di ogni gruppo
    for i, group in enumerate(groups):
        print(f"Gruppo {i + 1}:")
        for combination in group:
            mod_combination = [np.abs(x) for x in combination]
            print(mod_combination)
        print()"""
    
    return groups


def group_coefficients(image, pixel, zero):
    
    # Per ogni pixel mancante
    x, y = pixel
    
    group = []
    distances = []
    surrounding_coeff = [] 
    
    # Consideriamo i pixel adiacenti al pixel mancante
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx, ny = x + dx, y + dy
            #print("Intorno: ", abs(image[ny, nx]))
            #print("Pos: ", nx, " ", ny, "\n")

            # Verifichiamo che il pixel sia all'interno dell'immagine
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:

                # Se il pixel non è nella lista dei pixel mancanti, lo aggiungiamo al gruppo
                if np.abs(image[ny, nx]) >= zero and (nx, ny) != (x, y):
                    #print("Approvato Intorno: ", abs(image[ny, nx]),"\n")
                    group.append(image[ny, nx])

                    # Calcola la distanza euclidea del pixel mancante da questo pixel
                    distance = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
                    #print("Distanza: ", distance)
                    distances.append(distance)
            
    #print("Grandezza gruppo: ", len(group))

    # Se il gruppo è composto da più di 4 coefficienti noti
    if len(group) > 4:
        # Ordina i coefficienti per distanza dal pixel mancante e prendi solo i primi 4
        sorted_group = [x for _, x in sorted(zip(distances, group))][:4]
        surrounding_coeff = sorted_group
        #mod_combination = [np.abs(x) for x in sorted_group]
        #print("Migliori: ", mod_combination)

        #Realizza i 4 gruppi da 3 coefficienti con differenti combinazioni
        groups = divide_into_groups(sorted_group)

        #print("Totale gruppi: ", len(groups))

        return surrounding_coeff, groups
    elif len(group) == 4:
        surrounding_coeff = group
        groups = divide_into_groups(group)
        
        #print("Totale gruppi: ", len(groups))

        return surrounding_coeff, groups
    else:
        return [], [] #ritorna una lista vuota



def find_area_type(image, group, pixel, strategy):
  
  if strategy == "gradiente":
    # Calcolo del gradiente per ogni coefficiente
    gradienti = np.gradient(group)
    
    """# Calcolo del gradiente complessivo considerando tutte le componenti spettrali
    overall_gradient = np.sqrt(sum(np.square(grad) for grad in gradienti))
        
    print("grad: ", abs(np.max(overall_gradient)))"""
    
    # Se tutti i gradienti sono elevati, il gruppo è su un bordo/linea (True)
    if 0.07 < abs(np.max(gradienti)) < 0.4 :
      return True
    else:
      return False

  #elif strategy == "mappa":
    
    

def interpolate_non_border_group(group):

    # Calcola la media aritmetica dei coefficienti nel gruppo
    interpolated_value = sum(group) / len(group)
    return interpolated_value



def interpolate_border_group(groups):

    # Inizializza la lista dei migliori gruppi e il valore minimo della deviazione standard
    best_groups = []
    min_std_values = []

    # Calcola la deviazione standard per ogni gruppo di combinazioni
    for group in groups:
        std = np.std(group)
        min_std_values.append(std)
    
    # Seleziona i valori delle deviazioni standard minime
    min_std_indices = sorted(range(len(min_std_values)), key=lambda k: min_std_values[k])[:3]

    # Seleziona i gruppi corrispondenti ai valori minimi della deviazione standard
    for index in min_std_indices:
        best_groups.append(groups[index])
    
    # Inizializza il valore interpolato come zero
    interpolated_value = 0
    
    # Interpola il valore del pixel mancante utilizzando la media dei gruppi con deviazione standard minima
    for group in best_groups:
         for i in range(len(group[0])):  # Iteriamo su ogni elemento della tupla
            # Somma i valori reali corrispondenti in ciascuna tupla
            sum_values = sum(value[i].real for value in group)
            interpolated_value += sum_values / len(group)  # Calcoliamo la media
    
    return interpolated_value



def interpolation(image):

    #Valore dei pixel più vicini allo zero
    zero = 1e-2

    groups = []
    strategy = "gradiente"
    
    count1 = 0
    count2 = 0

    # Trova i pixel mancanti
    missing_pixels = find_missing_pixels(image, zero)
    print("n. Pixel == 0: ", len(missing_pixels))

    """print("Pixel: ", abs(image[767, 767]))
    surrounding_coeff, group = group_coefficients(image, [0,763], zero)

    groups.append(group) #op. non necessaria 
    print("Gruppi tornati: ", len(groups))

    if len(surrounding_coeff ) > 0 : 
        result = find_area_type(image, surrounding_coeff, [767, 767], strategy)
        print("Bordi: ", result)"""

    
    for missing_pixel in missing_pixels:

        x, y = missing_pixel

        #Trova i gruppi intorno ai pixel mancanti
        surrounding_coeff, group = group_coefficients(image, missing_pixel, zero)

        groups.append(group) #op. non necessaria 


        if len(surrounding_coeff) > 0:
            # Interpola
            # Se hai trovato un'area con dei bordi
            if find_area_type(image, surrounding_coeff, missing_pixel, strategy) :
                #deviazione standard
                count1 += 1
                if len(group) > 0:
                    interpolated_value = interpolate_border_group(group)
                else:
                    interpolated_value = image[y, x]
            else:
                #media
                count2 += 1
                interpolated_value = interpolate_non_border_group(surrounding_coeff)
        else:
            interpolated_value = image[y, x]

        # Assegna il valore interpolato al pixel mancante
    
        image[y, x] = interpolated_value

    print("\nGruppi: ", len(groups))
    print("\nBordi: ", count1)
    print("No Bordi: ", count2)

    return image
    
        



#--------------------MAIN-----------------------


# Carica un'immagine ad alta risoluzione
image_hr = cv2.imread("area_centrale.png", cv2.IMREAD_GRAYSCALE)
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

#HR_upsampling = up_sampling(HR_rotate, HR)
#print("\nSize ups: ", HR_rotate.shape)


plt.figure(figsize=(12, 8))
#plt.imshow(HR_upsampling, cmap='gray')
plt.title('HR Up-Sampling')
#plt.show()



# Calcola la trasformata curvelet discreta

HR_tras = fdct(HR_rotate)
print("\nReal Pixel[0][0] ", np.real(HR_tras[0][0]))
print("Imag Pixel[0][0] ", np.imag(HR_tras[0][0]))
print("Abs Pixel[0][0] ", np.abs(HR_tras[0][0]))

print("\nReal Pixel[0][768] ", np.real(HR_tras[0][768]))
print("Imag Pixel[0][768] ", np.imag(HR_tras[0][768]))
print("Abs Pixel[0][768] ", np.abs(HR_tras[0][768]))

x, y = HR_tras.shape
x = x-1
y = y-1
metax, metay = x//2, y//2
print("Meta: ", metax, metay)



# Interpolazione
HR_intrp = interpolation(HR_tras)



# Calcola l'inversa della trasformata delle curvelet per ricostruire l'immagine originale
FDCT_inv = HR_intrp.H @ c

# Verifica che l'immagine originale e quella ricostruita siano simili entro una certa tolleranza
#np.testing.assert_allclose(HR_upsampling, FDCT_inv, rtol=1e-6, atol=1e-8)
#np.testing.assert_allclose(HR_upsampling, FDCT_inv)

# Calcola l'errore quadratico medio (RMSE)
#rmse = np.sqrt(np.mean((HR_upsampling - FDCT_inv)**2))

# Stampa il RMSE
#print("RMSE:", rmse)

