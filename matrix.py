import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from curvelops import FDCT2D
from curvelops.plot import curveshow
from itertools import combinations, cycle
from curvelops import fdct2d_wrapper as ct
from joblib import Parallel, delayed



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
   
    print("Angles: ", len(c[3][0]))
    
    return c


# F - Interpolazione
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
    
    # Stampa gli elementi di ogni gruppo
    for i, group in enumerate(groups):
        print(f"Gruppo {i + 1}:")
        for combination in group:
            mod_combination = [np.abs(x) for x in combination]
            print(mod_combination)
        print()
    
    return groups



def group_coefficients(image, pixel, known_pixels):
    
    # Per ogni pixel mancante
    x, y = pixel
    
    print("\ncoord: ", x, " ", y)
    group = []
    surrounding_coeff = [] 
    
    # Consideriamo i pixel adiacenti al pixel mancante
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx, ny = x + dx, y + dy
            
            # Verifichiamo che il pixel sia all'interno dell'immagine
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                print("Pixel pos: ", nx, " ", ny)
            
                # Se il pixel non è nella lista dei pixel mancanti, lo aggiungiamo al gruppo
                if (nx, ny) in known_pixels:
                    print("Approvato Intorno: ", image[nx, ny],"\n")
                    group.append((nx, ny))

    surrounding_coeff = group[:]
            
    print("Grandezza gruppo: ", len(group))
    print("Surrounding : ", len(surrounding_coeff))

    # Se il gruppo è composto da più di 4 coefficienti noti
    if len(group) == 4:
        
        #print("Totale gruppi: ", len(groups))
        return surrounding_coeff, []

    elif len(group) < 4:
        
        while len(group) < 4:
            # Cerchiamo il pixel non aggiunto più vicino a (x, y)
            closest_pixel = calculate_distance(pixel, group, known_pixels)  

            # Esegue il calcolo parallelo della distanza per ogni pixel mancante
            #closest_pixel = parallelize_distance_calculation(pixel, group, known_pixels)
    
            # Aggiungiamo il pixel più vicino al gruppo e lo aggiungiamo alla lista dei pixel aggiunti
            group.append(closest_pixel)
            print("aggiunto pixel: ", closest_pixel)
        
        groups = divide_into_groups(group)
       
        return surrounding_coeff, groups 

        
 
def interpolate_non_border_group(group, fdct):

    sum  = 0

    # Calcola la media aritmetica dei coefficienti nel gruppo
    for pixel in group:
        print("fdct pixel: ", np.abs(fdct[pixel]))
        sum += fdct[pixel]

    interpolated_value = sum / len(group)
    print("Interp: ", np.abs(interpolated_value))
    return interpolated_value


def std_group(group, image):
    
    pixels = []
    print("Gr: ", group)
    for pixel in group[0]:
        print("Pixel:", pixel)
        x, y = pixel
        print("pixel ", np.abs(image[x, y]))
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
        print("std: ", std, "\n")
        min_std_values.append(std)
    
    # Seleziona i valori delle deviazioni standard minime
    min_std_indices = sorted(range(len(min_std_values)), key=lambda k: min_std_values[k])[:3]

    # Seleziona i gruppi corrispondenti ai valori minimi della deviazione standard
    for index in min_std_indices:
        print("Best: ", groups[index])
        best_groups.append(groups[index])

     
    sum_values = 0
    # Interpola il valore del pixel mancante utilizzando la media dei gruppi con deviazione standard minima
    for group in best_groups:
         for pixel in group[0]:  # Iteriamo su ogni elemento della tupla
            # Somma i valori corrispondenti in ciascuna tupla
            x,y = pixel
            print("Somma: ", fdct[x,y])
            sum_values += fdct[x, y]
    
    print("Sum value: ", (sum_values))
    interpolated_value = sum_values / 9  # Calcoliamo la media
    print("Interp value: ", (interpolated_value))
    return interpolated_value



def interpolation(fdct, img_rotate, missing_pixels, known_pixels):
        
    count1 = 0
    count2 = 0

    print("n. Pixel == 0: ", len(missing_pixels))

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
        print(f"Coord interp: {x}, {y}")
        print("Value not interp: ", (fdct[x, y]))
        fdct[x, y] = interpolated_value

    print("\nBordi: ", count1)
    print("No Bordi: ", count2)

    return fdct


# Inverse descrete transform
def ifdct(image, fdct, scale, angle):
    xinv = ct.fdct2d_inverse_wrap( *image.shape, scale, angle, False, fdct)
    return xinv



def interpolation_wrapper(args):
    fdct, img_rotate, missing_pixels, known_pixels = args
    return interpolation(fdct.copy(), img_rotate.copy(), missing_pixels.copy(), known_pixels.copy())


#-------------------------Algoritmo 2 ---------------------------------------------------
def main():

    # Matrice 4x4
    matrice4x4 = np.array([[1, 2, 3, 4], 
                            [5, 6, 7, 8], 
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]])

    # Matrice 6x6
    matrice = np.array([[1, 2, 3, 4, 5, 6],
                        [7, 8, 9, 10, 11, 12],
                        [13, 14, 15, 16, 17, 18], 
                        [19, 20, 21, 22, 23, 24], 
                        [25, 26, 27, 28, 29, 30], 
                        [31, 32, 33, 34, 35, 36]])

    # Matrice 1
    matrice1 = np.array([[1, 2],
                        [3, 4]])

    # Matrice 2
    matrice2 = np.array([[5, 6],
                        [7, 8]])

    matrice3 = np.array([[9, 10], 
                        [11, 12]])

    matrice4 = np.array([[13, 14], 
                        [15, 16]])

    # Stampa delle matrici
    print("Matrice 1:")
    print(matrice1)
    print("\nMatrice 2:")
    print(matrice2)

    # Calcola le coordinate xh, yh per la griglia ad alta risoluzione
    xh1, yh1 = np.meshgrid(np.arange(matrice1.shape[1]), np.arange(matrice1.shape[0]))
    xh2, yh2 = np.meshgrid(np.arange(matrice2.shape[1]), np.arange(matrice2.shape[0]))

    # Calcola le coordinate x1, y1, x2, y2 come specificato nell'equazione
    x1 = 2 * xh1 + 1
    y1 = 2 * yh1 + 1
    x2 = 2 * xh2
    y2 = 2 * yh2

    # Calcola le dimensioni della griglia Quincunx
    rows, cols = matrice1.shape

    # Inizializza l'immagine risultante
    H = np.zeros((2 * rows, 2 * cols), dtype=np.uint8)

    # Trova le coordinate dove i pixel sono rimasti uguali a zero

    # Posizionare i pixel a bassa risoluzione sulla griglia ad alta risoluzione
    H[y1, x1] = matrice2
    H[y2, x2] = matrice1

    # Calcolare l'immagine risultante (può essere una media, una somma, ecc.)
    result_image = H  

    zero_coordinates = np.argwhere(H == 0)

    print("Result:\n", result_image)

    rows, columns = result_image.shape


    new=np.zeros((rows*2,columns*2))

    for j in range(columns):
        for i in range(rows):
            new[i+j,columns-j+i]=result_image[j,i]
        

    print("Rotate: \n", new)


    scale = 4
    angles = 4
    coeff = fdct(new, scale, angles)
    matrix = len(coeff)-1

    missing_pixels = find_missing_pixels(H)
    known_pixels = find_known_pixels(H)
    print("Coordinate 0 nuove :\n ", missing_pixels)
    print("Coordinate !=0:\n ", known_pixels)


    #image_interp = interpolation(coeff[matrix][0], new, missing_pixels, known_pixels)
    # Esegui l'interpolazione dei pixel mancanti in parallelo
    interpolated_coefficients = Parallel(n_jobs=-1)(
    delayed(interpolation_wrapper)(args) for args in [(coeff[matrix][0], new, missing_pixels, known_pixels)]
    )
    


    print("Prova, ", interpolated_coefficients[0][3,7])
    coeff[matrix][0] = interpolated_coefficients[0]


    inverse = ifdct(new, coeff, scale, angles)

    plt.figure(figsize=(12, 8))
    plt.imshow(np.real(inverse), cmap='gray')
    plt.title('inv')
    #plt.show()

    #print("Inv: ", np.abs(inverse))
    print("Len: ", inverse.shape)

    print("Final befor rotate: ", (inverse[3,7]))

    r, c = inverse.shape 
    a_rotate = np.zeros((r//2,c//2))

    for j in range(c//2):
        for i in range(r//2):
            a_rotate[j, i] = np.abs(inverse[i+j,c//2-j+i])
        

    print("ARotate: \n", a_rotate)
    

if __name__ == "__main__":
    main()
        