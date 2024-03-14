import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from curvelops import fdct2d_wrapper as ct
from itertools import combinations, cycle
from joblib import Parallel, delayed


# Ottieni una porzione dell'immagine (16x16)
def cut_image(image):
    
    # Larghezza della porzione da estrarre
    dimensione_riquadro = 16

    x = 300
    y = 400

    # Estrai il riquadro desiderato
    new_img = image[y:y+dimensione_riquadro, x:x+dimensione_riquadro]


    return new_img


# Esegui lo slicing delle 2 sottoporzioni
def subimage(image):
    rows, columns =  image.shape

    hr_1 = image[0:rows, 0:columns]               # inizia da (0,0) e si estende fino a (768, 768)
    hr_2 = image[0:rows, 1:columns+1]
 
    return hr_1, hr_2



# A- Immagini LR 
def low_resolution(images_hr, M):
    images_lr = []
    for image_hr in images_hr:
        rows, columns =  image_hr.shape

        # Creazione di un'immagine vuota LR
        image_lr = np.zeros_like(image_hr[::M, ::M])

        # Iterazione su righe e colonne delle immagini HR di passo 2 (M)
        for i in range(0, rows, M):
            for j in range(0, columns, M):

                # Calcolo dell'indice corrispondente nelle immagini LR 
                k = i // M
                l = j // M

                # Calcolo della media nell'intorno MxM e assegnazione a lr_image
                image_lr[k, l] = int(image_hr[i:i+M, j:j+M].mean())

        # Aggiunta di lr_image alla lista di immagini LR
        images_lr.append(image_lr)

    return images_lr[0], images_lr[1]




#B - Campionamento Quincunx - Combinare frame I1 e I2 
def quincunx_sampling(I1, I2):
    

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
    new=np.zeros((rows*2,columns*2), dtype=image.dtype)
    
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

     
    # Inizializza la somma dei valori dei coefficienti per ogni gruppo
    sum_values_per_group = [0] * len(best_groups)   
    
    # Calcola la somma dei valori dei coefficienti per ogni gruppo
    for i, group in enumerate(best_groups):
        for pixel in group[0]:
            x, y = pixel
            sum_values_per_group[i] += fdct[x, y]
    
    # Calcola la media dei valori dei coefficienti per ogni gruppo
    interpolated_values = [sum_values / len(group[0]) for sum_values, group in zip(sum_values_per_group, best_groups)]
    
    # Restituisci la media delle medie dei valori interpolati per tutti i gruppi selezionati
    return sum(interpolated_values) / len(interpolated_values)

    """sum_values = 0
    # Interpola il valore del pixel mancante utilizzando la media dei gruppi con deviazione standard minima
    for group in best_groups:
         for pixel in group[0]:  # Iteriamo su ogni elemento della tupla
            # Somma i valori corrispondenti in ciascuna tupla
            x,y = pixel
            print("Somma: ", fdct[x,y])
            sum_values += fdct[x, y]
    
    interpolated_value = sum_values / 9  # Calcoliamo la media
    return interpolated_value"""

def interpolation(fdct, img_rotate, missing_pixels, known_pixels):
    fdct_copy = np.copy(fdct)

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
            interpolated_value = interpolate_border_group(group, fdct_copy)
        elif len(surrounding_coeff) == 4:
            #media
            count2 += 1
            interpolated_value = interpolate_non_border_group(surrounding_coeff, fdct_copy)
        
        # Assegna il valore interpolato al pixel mancante
        fdct_copy[x, y] = interpolated_value

    print("\nBordi: ", count1)
    print("No Bordi: ", count2)

    return fdct_copy
    
        
# FDCT inversa
def ifdct(image, fdct, scale, angle):
    xinv = ct.fdct2d_inverse_wrap( *image.shape, scale, angle, False, fdct)
    return xinv



def fill_missing_pixels(fdct_inverse, HR_rotate, missing_pixels):
    for x, y in missing_pixels:
        # Otteniamo il valore dalla trasformata inversa FDCT
        interpolated_value = fdct_inverse[x, y]
        # Assegniamo il valore alla posizione corrispondente in HR_rotate
        HR_rotate[x, y] = interpolated_value
    
    return HR_rotate


# A-Rotate
def a_rotate(image):
    r, c = image.shape 
    rows, columns = r//2, c//2
    a_rotate = np.zeros((rows, columns), dtype=image.dtype)

    for j in range(columns):
        for i in range(rows):
            a_rotate[j, i] = image[i+j,columns-j+i]

    return a_rotate


    
def interpolation_wrapper(args):
    fdct, img_rotate, missing_pixels, known_pixels = args
    return interpolation(fdct.copy(), img_rotate.copy(), missing_pixels.copy(), known_pixels.copy())



#--------------------MAIN-----------------------

def main():
    
    # Carica un'immagine ad alta risoluzione
    image_hr = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_hr, cmap='gray')
    plt.title('Image HR')
    plt.show()

    # Taglia una porzione più piccola
    image_cut = cut_image(image_hr)

    plt.figure(figsize=(12, 8))
    plt.imshow(image_cut, cmap='gray')
    plt.title('HR Cut')
    plt.show()
    print("Sub image size: ", image_cut.shape)

    # Ottieni le 2 sottoporzioni
    sub_images = subimage(image_cut)
    # Ottieni le due immagini LR
    I1, I2 = low_resolution(sub_images, 2)


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(I1, cmap='gray')
    plt.title('I1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(I2, cmap='gray')
    plt.title('I2')
    plt.axis('off')
    plt.show()

    print("\nI1 size: ", I1.shape)
    print("I2 size: ", I2.shape)


    #Combina i due frame LR
    HR = quincunx_sampling(I1, I2)
    plt.figure(figsize=(12, 8))
    plt.imshow(HR, cmap='gray')
    plt.title('Quincunx Sampling')
    plt.show()
    print("\nQuincunx size: ", HR.shape)

    print("I1 Pixel : \n", I1)
    print("I2 Pixel : \n", I2)
    print("HR Pixel : \n", HR)

    HR_rotate = rotate_quincunx_image(HR)
    plt.figure(figsize=(12, 8))
    plt.imshow(HR_rotate, cmap='gray')
    plt.title('HR Rotated')
    plt.show()

    print("\nRotate size: ", HR_rotate.shape)

    # Interpolazione
    missing_pixels = find_missing_pixels(HR)
    known_pixels = find_known_pixels(HR)

    scale = 4
    angles = 16

    coeff = fdct(HR_rotate, scale, angles)
    matrix = len(coeff)-1
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(coeff[matrix][0]), cmap='gray')
    plt.title('fdct')
    plt.show()

    

    image_interp = interpolation(coeff[matrix][0], HR_rotate, missing_pixels, known_pixels)
    coeff[matrix][0] = image_interp

    # Esegui l'interpolazione dei pixel mancanti in parallelo
    #interpolated_coefficients = Parallel(n_jobs=-1)(
    #delayed(interpolation_wrapper)(args) for args in [(coeff[matrix][0], HR_rotate, missing_pixels, known_pixels)]
    #)
    #coeff[matrix][0] = interpolated_coefficients[0]


    inverse = ifdct(HR_rotate, coeff, scale, angles)

    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(inverse), cmap='gray')
    plt.title('inv')
    plt.show()      

    fill_image = fill_missing_pixels(np.abs(inverse), HR_rotate, missing_pixels)
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(fill_image), cmap='gray')
    plt.title('fill')
    plt.show()

    final_image = a_rotate(fill_image)
    plt.figure(figsize=(12, 8))
    plt.imshow(final_image, cmap='gray')
    plt.title('final_image')
    plt.show()


if __name__ == "__main__":
    main()
