import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from curvelops import FDCT2D
from curvelops.plot import curveshow
from itertools import combinations, cycle

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
print("\nMatrice 3:")
print(matrice3)
print("\nMatrice 4:")
print(matrice4)

"""# Matrice 1 4x4
matrice1 = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])

# Matrice 2 4x4
matrice2 = np.array([[17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32]])"""


            

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

    
    print("Angles: ", len(coeff[3]))

    # Visualizza le curvelet
    fig_axes = curveshow(coeff, kwargs_imshow=dict(extent=[0, 1, 1, 0]))
    #plt.show()

    return coeff[3][0]




#D - Missing-Pixel
def find_missing_pixels(image):
     
    # TRova le coordinate uguali a zero dell'immagine prima della rotazione
    zero_coordinates = np.argwhere(image == 0)

    rows, columns = image.shape

    missing_pixels = []
    for coord in zero_coordinates:
        i, j = coord
        i_n = i + j
        j_n = columns - j + i
        missing_pixels.append((i_n, j_n))
          
    return missing_pixels


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



def group_coefficients(image, pixel, missing_pixels):
    
    # Per ogni pixel mancante
    x, y = pixel
    
    print("\ncoord: ", x, " ", y)
    group = []
    distances = []
    surrounding_coeff = [] 
    
    # Consideriamo i pixel adiacenti al pixel mancante
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx, ny = x + dx, y + dy
            
            # Verifichiamo che il pixel sia all'interno dell'immagine
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                print("Pixel pos: ", nx, " ", ny)
            
                # Se il pixel non è nella lista dei pixel mancanti, lo aggiungiamo al gruppo
                if (nx, ny) != (x, y):
                    print("Approvato Intorno: ", image[nx, ny],"\n")
                    group.append((nx, ny))

                    # Calcola la distanza euclidea del pixel mancante da questo pixel
                    distance = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
                    #print("Distanza: ", distance)
                    distances.append(distance)
            
    print("Grandezza gruppo: ", len(group))

    # Se il gruppo è composto da più di 4 coefficienti noti
    if len(group) > 4:

        # Ordina i coefficienti per distanza dal pixel mancante e prendi solo i primi 4
        sorted_group = [x for _, x in sorted(zip(distances, group), reverse=True)][:4]
        
        surrounding_coeff = sorted_group
        mod_combination = [x for x in sorted_group]
        print("Migliori: ", mod_combination)

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



def find_area_type(group, image):
  
    border = False
    for pixel in group:
        if image[pixel] == 0:
            border = True     
    
    return border


  #elif strategy == "mappa":


 
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



def interpolation(fdct, image, img_rotate):

    missing_pixels = find_missing_pixels(image)

    print("Coordinate 0 nuove :\n ", missing_pixels)
        
    groups = []
    strategy = "gradiente"
    
    count1 = 0
    count2 = 0

    print("n. Pixel == 0: ", len(missing_pixels))

    for missing_pixel in missing_pixels:

        x, y = missing_pixel

        #Trova i gruppi intorno ai pixel mancanti
        surrounding_coeff, group = group_coefficients(img_rotate, missing_pixel, missing_pixels)

        groups.append(group) #op. non necessaria 


        # Interpola
        # Se hai trovato un'area con dei bordi
        if find_area_type(surrounding_coeff, img_rotate) :
            #deviazione standard
            count1 += 1
            #interpolated_value = 1
            interpolated_value = interpolate_border_group(group, fdct)
        else:
            #media
            count2 += 1
            interpolated_value = interpolate_non_border_group(surrounding_coeff, fdct)
        

        # Assegna il valore interpolato al pixel mancante
        fdct[x, y] = interpolated_value

    print("\nGruppi: ", len(groups))
    print("\nBordi: ", count1)
    print("No Bordi: ", count2)

    return image


#-------------------------Algoritmo 2 ---------------------------------------------------
    
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
print("Coordinate 0:\n ", zero_coordinates)

rows, columns = result_image.shape



new=np.zeros((rows*2,columns*2))

for j in range(columns):
    for i in range(rows):
        new[i+j,columns-j+i]=result_image[j,i]
    

print("Rotate: \n", new)

coeff = fdct(new)


image_final = interpolation(coeff, H, new)


