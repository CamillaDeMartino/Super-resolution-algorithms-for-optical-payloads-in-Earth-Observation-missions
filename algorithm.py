import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from curvelops import fdct2d_wrapper as ct
from itertools import combinations, cycle
import multiprocessing

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
    # Passo di Up-sampling
    new=np.zeros((rows*2,columns*2))
    
    for j in range(columns):
        for i in range(rows):
            new[i+j,columns-j+i]=image[j,i]
    

    return new


#D - Missing-Pixel
def find_missing_pixels(image):

    rows, columns = image.shape

    # Trova le coordinate uguali a zero dell'immagine prima della rotazione
    missing_pixels = []
    for i in range(rows):
        for j in range(int(columns/2)):
            if np.remainder(i,2)==0:
                i_new, j_new = i, j*2+1
                missing_pixels.append((i_new + j_new, columns - i_new + j_new))

            if np.remainder(i,2)==1:
                i_new, j_new = i, j*2
                missing_pixels.append((i_new + j_new, columns - i_new + j_new))

          
    return missing_pixels


# Known-Pixel
def find_known_pixels(image):
    
    rows, columns = image.shape

    # Trova le coordinate diverse da zero dell'immagine prima della rotazione
    known_pixels = []
    for i in range(rows):
        for j in range(int(columns/2)):
            if np.remainder(i,2)==0:
                i_new, j_new = i, j*2
                known_pixels.append((i_new + j_new, columns - i_new + j_new))

            if np.remainder(i,2)==1:
                i_new, j_new = i, j*2+1
                known_pixels.append((i_new + j_new, columns - i_new + j_new))  

          
    return known_pixels

def calculate_distance(pixel, group, known_pixels):
    x,y = pixel
    min_distance = float('inf')  # Inizializziamo la distanza minima con un valore infinito
    closest_pixel = None  # Inizializziamo il pixel più vicino come None
    
    # Copia la lista dei known_pixels
    remaining_known_pixels = known_pixels.copy()
    # Rimuovi i pixel già presenti nel gruppo dalla lista dei known_pixels
    for group_pixel in group:
        if group_pixel in remaining_known_pixels:
            remaining_known_pixels.remove(group_pixel)

    for dx in range(-3, 4):
        for dy in range(-3, 4):
            nx, ny = x + dx, y + dy
            
            # Verifichiamo se il pixel non è nell'elenco dei pixel aggiunti e se è noto
            if (nx, ny) in remaining_known_pixels:
                
                # Calcoliamo la distanza euclidea dal pixel corrente a (x, y)
                distance = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
                # Se la distanza è minore della distanza minima attuale, aggiorniamo i valori
                if distance < min_distance:
                    min_distance = distance
                    closest_pixel = (nx, ny)

    return closest_pixel



#E - Discrete Curvelet Domain 
def fdct(image, scale, angles):

    print("Img dims: ", image.shape)

    c = ct.fdct2d_forward_wrap(scale, angles, False, image)
    print("\nLa lista coeff contiene", len(c), "elementi.")
    print("Dims c[0][0]: ", c[0][0].shape)
    print("Dims c[1][0]: ", c[1][0].shape)
    print("Dims c[2][0]: ", c[2][0].shape)
    print("Dims c[3][0]: ", c[3][0].shape)

    print("Angles: ", len(c[3][0]))

    return c




#F - Interpolation
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



def group_coefficients(image, pixel, known_pixels):
    
    # Per ogni pixel mancante
    x, y = pixel
    
    group = []
    surrounding_coeff = [] 
    
    # Consideriamo i pixel adiacenti al pixel mancante
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx, ny = x + dx, y + dy
            
            # Verifichiamo che il pixel sia all'interno dell'immagine
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                # Se il pixel non è nella lista dei pixel mancanti, lo aggiungiamo al gruppo
                if (nx, ny) in known_pixels:
                    group.append((nx, ny))

    surrounding_coeff = group[:]  

    # Se il gruppo è composto da più di 4 coefficienti noti
    if len(group) == 4:
        return surrounding_coeff, []

    elif len(group) < 4:     
        while len(group) < 4:
            # Cerchiamo il pixel non aggiunto più vicino a (x, y)           
            closest_pixel = calculate_distance(pixel, group, known_pixels)
            
            # Aggiungiamo il pixel più vicino al gruppo e lo aggiungiamo alla lista dei pixel aggiunti
            group.append(closest_pixel)            
        
        groups = divide_into_groups(group)
       
        return surrounding_coeff, groups 
    

def interpolate_non_border_group(group, fdct):

    sum  = 0

    # Calcola la somma dei coefficienti nel gruppo identificandoli in base alle coordinate nell'immagine trasformata
    for pixel in group:
        sum += fdct[pixel]

    # Calcola la media
    interpolated_value = sum / len(group)
    return interpolated_value


def std_group(group, image):
    
    pixels = []
    for pixel in group[0]:
        x, y = pixel
        pixels.append(image[x, y])
    
    
    std = np.std(pixels)
    return std


def interpolate_border_group(groups, fdct):

    # Inizializza la lista dei migliori gruppi e il valore minimo della deviazione standard
    best_groups = []
    min_std_values = []

    # Calcola la deviazione standard per ogni gruppo di combinazioni
    for group in groups:
        std = std_group(group, fdct)
        min_std_values.append(std)
    
    # Seleziona i valori delle deviazioni standard minime
    min_std_indices = sorted(range(len(min_std_values)), key=lambda k: min_std_values[k])[:3]

    # Seleziona i gruppi corrispondenti ai valori minimi della deviazione standard
    for index in min_std_indices:
        best_groups.append(groups[index])

     
    sum_values = 0
    # Interpola il valore del pixel mancante utilizzando la media dei gruppi con deviazione standard minima
    for group in best_groups:
         for pixel in group[0]:  # Iteriamo su ogni elemento della tupla
            # Somma i valori corrispondenti in ciascuna tupla
            x,y = pixel
            sum_values += fdct[x, y]
    
    interpolated_value = sum_values / 9  # Calcoliamo la media
    return interpolated_value



def interpolation(fdct, img_rotate, missing_pixels, known_pixels):


    count1 = 0
    count2 = 0

    for missing_pixel in missing_pixels:
        
        x, y = missing_pixel

        #Trova i gruppi intorno ai pixel mancanti
        surrounding_coeff, group = group_coefficients(img_rotate, missing_pixel, known_pixels)

        # Interpola
        # Se hai trovato un'area con dei bordi
        if len(surrounding_coeff) < 4:
            #deviazione standard
            count1 += 1
            interpolated_value = interpolate_border_group(group, fdct)
        elif len(surrounding_coeff) == 4:
            #media
            count2 += 1
            interpolated_value = interpolate_non_border_group(surrounding_coeff, fdct)
        

        # Assegna il valore interpolato al pixel mancante
        fdct[x, y] = interpolated_value

    print("\nBordi: ", count1)
    print("No Bordi: ", count2)

    return fdct
    
        
# FDCT inversa
def ifdct(image, fdct, scale, angle):
    xinv = ct.fdct2d_inverse_wrap( *image.shape, scale, angle, False, fdct)
    return xinv



# A-Rotate
def a_rotate(image):
    r, c = image.shape 
    rows, columns = r//2, c//2
    a_rotate =np.zeros((rows, columns))

    for j in range(columns):
        for i in range(rows):
            a_rotate[j, i] = image[i+j,columns-j+i]
    



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


# Interpolazione
missing_pixels = find_missing_pixels(HR)
known_pixels = find_known_pixels(HR)

scale = 4
angles = 4
coeff = fdct(HR_rotate, scale, angles)
matrix = len(coeff)-1

#image_interp = interpolation(coeff[matrix][0], HR_rotate, missing_pixels, known_pixels)
# Esegui l'interpolazione dei pixel mancanti in parallelo
pool = multiprocessing.Pool()
interpolated_coefficients = pool.starmap(interpolation, [(coeff[matrix][0], HR_rotate, missing_pixels, known_pixels)])
pool.close()
pool.join()
coeff[matrix][0] = interpolated_coefficients[0]

#coeff[matrix][0] = image_interp

inverse = ifdct(HR_rotate, coeff, scale, angles)

plt.figure(figsize=(12, 8))
plt.imshow(np.abs(inverse), cmap='gray')
plt.title('inv')
plt.show()

final_image = a_rotate(np.abs(inverse))
