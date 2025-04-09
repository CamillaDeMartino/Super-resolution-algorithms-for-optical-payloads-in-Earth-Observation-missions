import cv2
import numpy as np
from matplotlib import pyplot as plt
from curvelops import fdct2d_wrapper as ct
from itertools import combinations, cycle
from joblib import Parallel, delayed
import glymur as gly 




# A- Immagini LR 
def binning4x4(image):
   
    n,m = image.shape[:2]
    offset_y = 3
    offset_x = 2

    # Riduciamo le dimensioni dell'immagine per il binning
    new_height = n // 4 + (1 if n % 4 != 0 else 0)
    new_width = m // 4 + (1 if m % 4 != 0 else 0)

    new_height2 = (n - offset_y) // 4 + (1 if (n - offset_y) % 4 != 0 else 0)
    new_width2 = (m -offset_x) // 4 + (1 if (m - offset_x) % 4 != 0 else 0)

    #dtype=np.dtype(matrix)
    low_img1 = np.zeros((new_height, new_width))
    low_img2 = np.zeros((new_height2, new_width2))

    
    for i in range(0, n, 4):
        for j in range(0, m, 4):
            block1 = image[i: min(i+4, n), j:min(j+4, m)]
            low_img1[i//4, j//4] = np.mean(block1)

            # Seleziona il blocco con offset (gestione bordi)
            i_offset = i + offset_y
            j_offset = j + offset_x
            if i_offset < n and j_offset < m:
                block2 = image[i_offset:min(i_offset+4, n), j_offset:min(j_offset+4, m)]
                low_img2[i//4, j//4] = np.mean(block2)

    return low_img1, low_img2

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
    rows, cols = I1.shape[:2]

    # Inizializza l'immagine risultante
    H = np.zeros((2 * rows, 2 * cols), dtype=np.uint16)

    # Posizionare i pixel a bassa risoluzione sulla griglia ad alta risoluzione
    H[y1, x1] = I2
    H[y2, x2] = I1

    #Trasformazione = invertire le matrici???
    result_image = H  

    return result_image




#C - Rotate image by 45 degree
def rotate_quincunx_image(image):

    rows, columns = image.shape[:2]
    # Passo di Up-sampling
    #new=np.zeros((rows*2,columns*2), dtype=image.dtype)
    new=np.zeros((rows*2,columns*2), dtype=np.uint16)

    for j in range(columns):
        for i in range(rows):
            new[i+j,columns-j+i]=image[j,i] ##
    

    return new


#D - Missing-Pixel
def find_missing_pixels(image):

    rows, columns = image.shape[:2]

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
    
    rows, columns = image.shape[:2]

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
                if x == 1 and y == 65:
                    print("Pixel pos: ", nx, " ", ny)

                # Se il pixel non è nella lista dei pixel mancanti, lo aggiungiamo al gruppo
                if (nx, ny) in known_pixels:
                    if x == 1 and y == 65:
                        print("Approvato val: ", image[nx, ny],"\n")
                    group.append((nx, ny))

    surrounding_coeff = group[:]
    if x == 1 and y == 65:
        print("Grandezza gruppo: ", len(group))
        print("Surrounding : ", len(surrounding_coeff))

    # Se il gruppo è composto da 4 coefficienti noti allora non abbiamo bisogno di trovarne altri
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
        #print("fdct pixel: ", fdct[pixel])
        sum += fdct[pixel]

    # Calcola la media
    interpolated_value = sum / len(group)
    #print("Interp: ", interpolated_value)

    return interpolated_value


def std_group(group, image, missing_pixel):
    x, y = missing_pixel
   
    pixels = []
    if x == 1 and y == 65:
        print("Gr: ", group)
    for pixel in group[0]:
        x, y = pixel
        pixels.append(image[x, y])
    
    
    std = np.std(pixels)
    return std


def interpolate_border_group(groups, fdct, missing_pixel):
    x, y = missing_pixel
    # Inizializza la lista dei migliori gruppi e il valore minimo della deviazione standard
    best_groups = []
    min_std_values = []

    # Calcola la deviazione standard per ogni gruppo di combinazioni
    for group in groups:
        std = std_group(group, fdct, missing_pixel)
        if x == 1 and y == 65:
            print("std: ", std)
        min_std_values.append(std)
    
    # Seleziona i valori delle deviazioni standard minime
    min_std_indices = sorted(range(len(min_std_values)), key=lambda k: min_std_values[k])[:3]

    # Seleziona i gruppi corrispondenti ai valori minimi della deviazione standard
    for index in min_std_indices:
        if x == 1 and y == 65:
            print("Best groups: ", groups[index])
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
    interpolated_value = sum(interpolated_values) / len(interpolated_values)


    return interpolated_value

    # Altro metodo di interpolazione
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
            #print("Border Pixel")
        
            #deviazione standard
            count1 += 1
            interpolated_value = interpolate_border_group(group, fdct_copy, missing_pixel)
            
        elif len(surrounding_coeff) == 4:
            #media
            #print("No Border Pixel")
            count2 += 1
            interpolated_value = interpolate_non_border_group(surrounding_coeff, fdct_copy)
        
        # Assegna il valore interpolato al pixel mancante
        fdct_copy[x, y] = interpolated_value

    print("\nBordi: ", count1)
    print("No Bordi: ", count2)

    return fdct_copy
    
        
# FDCT inversa
def ifdct(image, fdct, scale, angle):
    matrix = len(fdct) - 1
    print("IM: ", image.shape)
    xinv = ct.fdct2d_inverse_wrap( *image.shape, scale, angle, False, fdct)

    return xinv



# A-Rotate
def a_rotate(image):
    r, c = image.shape[:2]
    rows, columns = r//2, c//2
    #a_rotate = np.zeros((rows, columns), dtype=image.dtype)
    a_rotate = np.zeros((rows, columns), dtype=np.uint16)

    for j in range(columns):
        for i in range(rows):
            a_rotate[j, i] = image[i+j,columns-j+i]

    return a_rotate


    
def interpolation_wrapper(args):
    fdct, img_rotate, missing_pixels, known_pixels = args
    return interpolation(fdct.copy(), img_rotate.copy(), missing_pixels.copy(), known_pixels.copy())



# Restored
def down_sampling2(image):
   
    # Fattore down-sampling 
    f = 2

    m,n = image.shape[:2]
    m_new = int(m/f)
    n_new = int(n/f)

    # Crea matrice di zeri per la matrice di sovracampionamento 
    img_downsampled1 = np.zeros((m_new, n_new), dtype=np.uint16) 
    img_downsampled2 = np.zeros((m_new, n_new), dtype=np.uint16) 
    for i in range(m_new): 
        for j in range(n_new): 
            img_downsampled1[i, j] = int(image[2*i:2*i+1, 2*j:2*j+1].mean())
            img_downsampled2[i, j] = int(image[2*i+1:2*i+2, 2*j+1:2*j+2].mean())


    return img_downsampled1

def scale_DN(low_img,restored_image):
    x,y = low_img.shape[:2]
    
    ratios = np.zeros((x,y))
    lowa, lowb = binning4x4(restored_image)

    # Upsample lowa per allinearlo con low_img
    lowa_upsampled = np.kron(lowa, np.ones((2, 2)))  # Aumenta la dimensione a 16x16

    ratios = low_img/lowa_upsampled
    print(ratios)
    upsampled_ratios = np.kron(ratios, np.ones((2, 2)))
    rescaled = upsampled_ratios*restored_image
    return rescaled



#Compressione e decompressione
def co_decompression(image1, image2, factor):
    
    #File1 - compression
    name_file1='LR1_4.jp2'
    jp3 = gly.Jp2k(name_file1, data=np.uint16(image1), cratios=[factor])
    
    #File2 - compression
    name_file2='LR2_4.jp2'
    jp2 = gly.Jp2k(name_file2, data=np.uint16(image2), cratios=[factor])


    # Decompression
    jp1_3 = gly.Jp2k(name_file1)
    jp2_3 = gly.Jp2k(name_file2)


    fullres1 = jp1_3[:]
    fullres2 = jp2_3[:]

    return fullres1, fullres2

#--------------------MAIN-----------------------

def main():
    
    # Carica un'immagine ad alta risoluzione
    image_hr = cv2.imread("/home/camilla/Scrivania/Tesi/Images/Vegetation_Apr_20200409_B4.tif", cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_hr, cmap='viridis')
    plt.colorbar()
    plt.title('Image HR')
    plt.show()

    # Porzione più piccola dell'immagine originale
    image_cut = image_hr[64:128, 64:128]

    plt.figure(figsize=(12, 8))
    plt.imshow(image_cut, cmap='viridis')
    plt.colorbar()
    plt.title('HR Cut')
    plt.show()
    print("Sub image size: ", image_cut.shape[:2])


    # Ottieni le due immagini LR (2 metodi possibili)
    #I1, I2 = low_resolution(sub_images, 2)
    I1, I2 = binning4x4(image_hr[64:128, 64:128])

    #I1, I2 = co_decompression(I1, I2, 2)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(I1, cmap='viridis')
    plt.colorbar()
    plt.title('I1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(I2, cmap='viridis')
    plt.colorbar()
    plt.title('I2')
    plt.axis('off')
    plt.show()

    print("\nI1 size: ", I1.shape[:2])
    print("I2 size: ", I2.shape[:2])


    #Combina i due frame LR
    HR = quincunx_sampling(I1, I2)
    plt.figure(figsize=(12, 8))
    plt.imshow(HR, cmap='viridis')
    plt.colorbar()
    plt.title('Quincunx Sampling')
    plt.show()
    print("\nQuincunx size: ", HR.shape[:2])

    print("I1 Pixel : \n", I1)
    print("I2 Pixel : \n", I2)
    print("HR Pixel : \n", HR)

    HR_rotate = rotate_quincunx_image(HR)

    plt.figure(figsize=(12, 8))
    plt.imshow(HR_rotate, cmap='viridis')
    plt.colorbar()
    plt.title('HR Rotated')
    plt.show()

    print("\nRotate size: ", HR_rotate.shape[:2])
    #print("R:\n", HR_rotate)

    # Interpolazione
    missing_pixels = find_missing_pixels(HR)
    known_pixels = find_known_pixels(HR)

    scale = 16
    angles = 64

    coeff = fdct(HR_rotate, scale, angles)
    matrix = len(coeff)-1
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(coeff[matrix][0]), cmap='viridis')
    plt.colorbar()
    plt.title('fdct')
    plt.show()

    
    image_interp = interpolation(coeff[matrix][0].copy(), HR_rotate.copy(), missing_pixels, known_pixels)
    coeff[matrix][0] = image_interp


    # Esegui l'interpolazione dei pixel mancanti in parallelo (se necessario)
    #interpolated_coefficients = Parallel(n_jobs=-1)(
    #delayed(interpolation_wrapper)(args) for args in [(coeff[matrix][0], HR_rotate, missing_pixels, known_pixels)]
    #)
    #coeff[matrix][0] = interpolated_coefficients[0]

    inverse = ifdct(HR_rotate, coeff.copy(), scale, angles)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(inverse), cmap='viridis')
    plt.colorbar()
    plt.title('inv')
    #plt.show()      

    

    final_image = a_rotate(np.abs(inverse))
    #plt.figure(figsize=(12, 8))
    #plt.imshow(final_image, cmap='gray')
    #plt.title('final_image')
    #plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(final_image, cmap='viridis')
    plt.colorbar()
    plt.title('Final')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_cut, cmap='viridis')
    plt.colorbar()
    plt.title('Original')
    plt.axis('off')
    #plt.show()

    #image_restored = restore(I1, final_image, 2)
    image_restored = scale_DN(I1, final_image)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_restored, cmap='viridis')
    plt.colorbar()
    plt.title('Restored Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_cut, cmap='viridis')
    plt.colorbar()
    plt.title('Original')
    plt.axis('off')
    plt.show()
    

if __name__ == "__main__":
    main()
