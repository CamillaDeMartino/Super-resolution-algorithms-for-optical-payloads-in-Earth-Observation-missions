import cv2 
import numpy as np
from matplotlib import pyplot as plt
from wand.image import Image

# Ottieni immagine centrata in 769x769
def img_center(image):
    altezza, larghezza = image.shape[:2]
    x_inizio = (larghezza - 769) // 2
    y_inizio = (altezza - 769) // 2
    x_fine = x_inizio + 769
    y_fine = y_inizio + 769

    center = image[y_inizio:y_fine, x_inizio:x_fine]

    return center


# Esegui lo slicing delle quattro sottoporzioni
def subimage(image):

    hr_1 = image[0:768, 0:768]               # inizia da (0,0) e si estende fino a (768, 768)
    hr_2 = image[0:768, 1:769]               # inizia da (0,1) e si estende fino a (768, 769)
    hr_3 = image[1:769, 0:768]
    hr_4 = image[1:769, 1:769]
 
    return hr_1, hr_2, hr_3, hr_4



# Immagini LR 
def low_resolution(images_hr, M):
    images_lr = []
    for image_hr in images_hr:
        # Creazione di un'immagine vuota LR
        image_lr = np.zeros_like(image_hr[::M, ::M])

        # Iterazione su righe e colonne delle immagini HR di passo 2 (M)
        for i in range(0, 768, M):
            for j in range(0, 768, M):

                # Calcolo dell'indice corrispondente nelle immagini LR 
                k = i // M
                l = j // M

                # Calcolo della media nell'intorno MxM e assegnazione a lr_image
                image_lr[k, l] = int(image_hr[i:i+M, j:j+M].mean())

        # Aggiunta di lr_image alla lista di immagini LR
        images_lr.append(image_lr)

    return images_lr

# Aggiungi rumore alle immagini LR
def add_noise(images):
    images_noise = []
    for image in images:
        
        # Converti l'immagine in oggetto Wand Image
        img = Image.from_array(image)

        # Applica il rumore gaussiano
        img.noise(noise_type='gaussian', attenuate=5)

        # Converte l'immagine di Wand Image a numpy array
        noisy_image = np.array(img)

        images_noise.append(noisy_image)
        
    return images_noise


# Estrai i punti chiave e i descrittori
def sift_features(images):

    features_list = []

    for idx, image in enumerate(images, 1):

        # Crea un oggetto SIFT
        sift = cv2.SIFT_create()

        keypoints, descriptors = sift.detectAndCompute(image, None)    # funzione che permette di calcolare punti chiave e i loro descrittori
                                                                       # None -> nessuna maschera, ma l'intera immagine
    
        # Salva i risultati nel dizionario
        features_dict = {
            'image': idx,
            'keypoints': keypoints,
            'descriptors': descriptors
        }

        features_list.append(features_dict)
        
    return features_list

    
# Le coppie sono tuple di due elementi, e x[0] rappresenta il primo elemento di ciascuna coppia, 
# che è un oggetto che ha un attributo .distance rappresentante la distanza della corrispondenza. 
# In questo modo, ordiniamo le coppie in base alla distanza.
def get_distance(x):
    return x[0].distance


# calcola la media degli angoli di inclinazione delle coppie cioè l'angolo di riferimento 
def angle_reference(matches, features):
    angles = []

    for match in matches:       # per ogni coppia di tuple trovate da FLANN
        i = match[0].queryIdx   # prendi l'indice il primo elemento (prima immagine)
        j = match[0].trainIdx   # prendi l'indice il secondo elemento (secondo elemento)

        # estrai le coordinate (pt)
        x_i1, y_i1 = features[0]['keypoints'][i].pt
        x_i2, y_i2 = features[1]['keypoints'][j].pt

        angles.append(np.arctan2(y_i2 - y_i1, x_i2 - x_i1))

    reference_angle = np.mean(angles)
    
    # Calcola T2 come media delle differenze assolute tra angolo di riferimento e gli angoli di inclinazione
    T2 = np.mean(np.abs(reference_angle - angles)) 

    return reference_angle, T2



# Elimina coppie se l'errore tra gli angli e l'angolo di riferimenro supera T2
def filter_matches(matches, features):

    reference_angle, T2 = angle_reference(matches, features)

    filtered_matches = []
    for match in matches:
        i = match[0].queryIdx
        j = match[0].trainIdx

        x_i1, y_i1 = features[0]['keypoints'][i].pt
        x_i2, y_i2 = features[1]['keypoints'][j].pt

        angle = np.arctan2(y_i2 - y_i1, x_i2 - x_i1)

        
        # Verifica l'errore dell'angolo rispetto all'angolo di riferimento 
        if np.abs(angle - reference_angle) < T2:
            filtered_matches.append(match)

    return filtered_matches



# Algoritmo di FLANN
def flann_matching(features):
    
    FLANN_INDEX_KDTREE = 0                                          # costante che specifica l'algoritmo di ricerca utilizzato da FLANN
                                                                    # K-D Tree per costruisce l'indice e per effettuare la ricerca approssimativa dei vicini più prossimi
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)                                 # FLANN interromperà la ricerca se controlla più di 50 nodi nell'albero

    good_matches_list = []

    for i in range(len(features) - 1):
        des1 = features[i]['descriptors']
        des2 = features[i + 1]['descriptors']

        # FLANN Matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)      #oggetto matcher, meccanismo per eseguire la corrispondenza

        # Trova i matching
        matches = flann.knnMatch(des1, des2, k=2)                       # desc1 e des2 sono i descrittori dei punti chiave nelle due immagini. 
                                                                        # Il parametro k indica il numero di vicini più prossimi da cercare. 
                                                                        # k=2 restituirà i due migliori match per ogni descrittore.
        
        # Ordina le coppie in base alla distanza di corrispondenza
        matches = sorted(matches, key=get_distance)             
        print(f"Matches coppia {i+1}-{i+2}: ", len(matches))


        good_match = filter_matches(matches, features[i:i+2])
        good_matches_list.append(good_match)

    return good_matches_list



# Filtra la distanza tra i pixel delle immagini LR
def filter_distance(images_lr):
    
    features = sift_features(images_lr)
    matches = flann_matching(features)
    total_distances = []
    
    print("\n")

    for i, match in enumerate(matches):
        if i < len(images_lr)-1:
            keypoints1 = features[i]['keypoints']
            
            keypoints2 = features[i+1]['keypoints']

            filter_distances, distances = calculate_distance(keypoints1, keypoints2, match)
            distances_images.append(filter_distances)
            
            total_distances.append(distances)
            print(f"Numero Pixel filtrati per distanza della coppia {i+1}: ", len(distances_images[i]))
            print(f"Distanza media filtrata tra pixel della coppia {i+1}: ", np.mean(distances_images[i]))
            print(f"Distanza media tra pixel non filtrata della coppia {i+1}: ", np.mean(total_distances[i]), "\n")



# Calcola la distanza tra i pixel delle immagini LR ottenute
def calculate_distance(keypoints1, keypoints2, matches):
    filter_distances = []
    distances = []
    
    T1 = 1    

    for match in matches:
        
        i = match[0].queryIdx
        j = match[0].trainIdx

        pt1 = keypoints1[i].pt
        pt2 = keypoints2[j].pt

        distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

        #Quanto la distanza ottenuta dista da quella ideale
        distances.append(abs(distance - 0.5))
       
        # Confronta la differenza tra la distanza e 0.5 con T1
        # lo spostamento desiderato è 0.5

        if abs(distance - 0.5) < T1:          
            filter_distances.append(distance)
    
    return filter_distances, distances



def mapping(images):

    # Dimensioni della matrice sovrapposta
    righe_sovrapposte = images[0].shape[0] * 2 + 1
    colonne_sovrapposte = images[0].shape[1] * 2 + 1

    # Creazione della matrice sovrapposta con zeri
    matrice_sovrapposta = np.zeros((righe_sovrapposte, colonne_sovrapposte))

    matrice1 = images[0]
    matrice2 = images[1]
    matrice3 = images[2]
    matrice4 = images[3]


    for i in range(righe_sovrapposte):
        for j in range(colonne_sovrapposte):
            if i == 0 and j == 0:
                matrice_sovrapposta[i, j] = np.squeeze(matrice1[i // 2, j // 2])
            elif i == 0 and j == colonne_sovrapposte-1:
                matrice_sovrapposta[i, j] = np.squeeze(matrice2[i, (j-1) // 2]) 
            elif i == righe_sovrapposte -1 and j == colonne_sovrapposte-1:
                matrice_sovrapposta[i, j] = np.squeeze(matrice4[(i-1) // 2, (j-1) // 2]) 
            elif i == righe_sovrapposte-1 and j == 0:
                matrice_sovrapposta[i, j] = np.squeeze(matrice3[(i-1) // 2, j // 2])
            elif i == 0:
                matrice_sovrapposta[i, j] = np.sum([matrice1[i// 2, (j-1) // 2], matrice2[i // 2, j // 2]])
            elif j == 0:
                matrice_sovrapposta[i, j] = np.sum([matrice1[(i-1)// 2, j// 2], matrice3[i // 2, j // 2]])
            elif j == colonne_sovrapposte-1:
                matrice_sovrapposta[i, j] = np.sum([matrice2[i // 2, (j-1) // 2], matrice4[(i-1) // 2, (j-1) // 2]]) 
            elif i == righe_sovrapposte-1:
                matrice_sovrapposta[i, j] = np.sum([matrice3[(i-1) // 2, j // 2], matrice4[(i-1) // 2, (j-1) // 2]])
            else:
                matrice_sovrapposta[i, j] = np.sum([matrice1[i//2, j // 2], matrice2[i // 2, (j-1) // 2], matrice3[(i-1) // 2, j // 2], matrice4[(i-1)//2, (j-1) // 2]])

    return matrice_sovrapposta



def gridHR(matrice_sovrapposta):

    # Dimensioni della matrice sovrapposta
    righe_sovrapposte = matrice_sovrapposta.shape[0] 
    colonne_sovrapposte = matrice_sovrapposta.shape[1]

    #Crea matrice finale HR:
    HR = np.zeros((righe_sovrapposte -1, colonne_sovrapposte-1))
    for i in range(righe_sovrapposte-1):
        for j in range(colonne_sovrapposte-1):
            HR[i, j] = (matrice_sovrapposta[i+1, j+1])/4

    return HR




# -------------------------- MAIN ----------------------------------------------------------------
# IMAGES SELECTION BASED ON SFME ------------------------------------------------------------------

# Carica un'immagine ad alta risoluzione
image_hr = cv2.imread("/home/camilla/Scrivania/Tesi/Images/test_image.png", cv2.IMREAD_GRAYSCALE)

# Ritaglia il centro dell'immagine 769x769
center = img_center(image_hr)

plt.figure(figsize=(12, 8))

# Immagine originale
plt.subplot(121)
plt.title("Immagine HR (850x850)")
plt.imshow(image_hr, cmap='gray')

# Immagine tagliata
plt.subplot(122)
plt.title("Immagine HR (769x769)")
plt.imshow(center, cmap='gray')

plt.show()

# 4 immagini hr 768x768
images_hr = subimage(center)

plt.figure(figsize=(12, 8))
if images_hr is not None:
    for i, image_hr in enumerate(images_hr, 1):
        plt.subplot(2, 2, i)
        plt.imshow(image_hr, cmap='gray')
        plt.title(f"HR_{i} (768x 768)")

plt.show()


# Size dell'intorno [MxM] del pixel in cui effettuare la media dei valori
M = 2         

# Simula immagini a bassa risoluzione
images_lr = low_resolution(images_hr, M)
plt.figure(figsize=(12, 8))
if images_lr is not None:
    for i, image_lr in enumerate(images_lr, 1):
        plt.subplot(2, 2, i)
        plt.imshow(image_lr, cmap='gray')
        plt.title(f"LR_{i} (384x384)")

plt.show()



# Aggiungi rumore alle immagini LR
noisy_images_lr = add_noise(images_lr) 

# Visualizzazione delle immagini LR con il rumore
plt.figure(figsize=(12, 8))
if noisy_images_lr is not None:
    for i, noisy_image_lr in enumerate(noisy_images_lr, 1):
        plt.subplot(2, 2, i)
        plt.imshow(noisy_image_lr, cmap='gray')
        plt.title(f"Noisy LR_{i} (384x384)")

plt.show()


# Estrai caratteristiche SIFT dall'immagine ad alta risoluzione
features = sift_features(noisy_images_lr)

# Applica FLANN
matches_flann = flann_matching(features)

distances_images = []
total_distances = []
print("\n")

# Visualizza i matching
for i, matches in enumerate(matches_flann):
    if i < len(noisy_images_lr)-1:
        print(f"Numero di matches filtrati tra immagine {i+1} e immagine {i+2}: {len(matches)}")

        image1 = noisy_images_lr[i]
        keypoints1 = features[i]['keypoints']
        
        image2 = noisy_images_lr[i+1]
        keypoints2 = features[i+1]['keypoints']

        
        # Frecce
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=None,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        )

        min_display = 100 #visualizza alcuni match
        img_matches = cv2.drawMatchesKnn( image1, keypoints1, image2, keypoints2, matches[:min_display], None, **draw_params)

        filter_distances, distances = calculate_distance(keypoints1, keypoints2, matches)
        distances_images.append(filter_distances)
        total_distances.append(distances)

        print(f"Numero Pixel filtrati per distanza della coppia {i+1}: ", len(distances_images[i]))
        print(f"Distanza media filtrata tra pixel della coppia {i+1}: ", np.mean(distances_images[i]))
        print(f"Distanza media tra pixel non filtrata della coppia {i+1}: ", np.mean(total_distances[i]), "\n")

        # Visualizza l'immagine con i match
        plt.figure(figsize=(12, 8))
        plt.imshow(img_matches)
        plt.show()

print("\n")

#filter_distance(images_lr)

#MULTI-FRAME IMAGES RECONSTRUCTION ----------------------------------------------------------------------------------------

image = mapping(noisy_images_lr)


result_image = gridHR(image)
print("Shape: ", result_image.shape) 


cv2.imwrite('/home/camilla/Scrivania/Tesi/Images/immagine_risultante.png', result_image)

# Visualizza l'immagine
plt.figure(figsize=(12, 8))
plt.imshow(result_image, cmap='gray')
plt.title('Immagine risultante')
plt.show()