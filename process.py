import matplotlib.pyplot as plt

from PIL import Image               # Librería para el procesamiento de imágenes
import matplotlib.image as img      # Para poder representar las imágenes más fácil  
# import cv2
# from skimage.util import random_noise
import numpy as np
import random as rd

# No funciona muy bien aun que no descarto mejorarlo para implementar la función 
# def add_noise(name:str,mode:str):
#     '''
#     # Explicación
#     Le damos una imagen de entrada y le mete ruido.
#     # Parámetros de entrada
#     name: nombre de la imagen a la que queremos meterle ruido
#     mode: tipo de ruido a añadir
#         - 'gaussian'-> Gaussian-distributed additive noise.
#         - 'localvar'-> Gaussian-distributed additive noise, with specified local variance at each point of image.
#         - 'poisson' -> Poisson-distributed noise generated from the data.
#         - 'pepper'  -> Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).
#         - 'salt'    -> Replaces random pixels with 1.
#         - 's&p'     -> Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
#         - 'speckle' -> Multiplicative noise using out = image + n * image, where n is Gaussian noise with specified mean & variance.
#     # Parámetros de salida
#     name2: nos devuelve el nombre de la imagen con ruido
#     '''
#     # img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
#     image = Image.open(name)
#     img_gauss = random_noise(image, mode = mode, mean = 0, var = 0.075)
#     img_gauss = (img_gauss*255).astype('uint8')
#     (thresh, img_gauss_bw) = cv2.threshold(img_gauss,128,255, cv2.THRESH_BINARY)
#     # img_gauss_bw= cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)
#     name2  = name[:name.find('.')]+'_noise.png' # Le añado un _noise a las imagenes con ruido
#     cv2.imwrite(name2,img_gauss_bw)
#     return img_gauss_bw # Nos devuele la imagen como una matriz 

def process_image(name:str,size:tuple):
    '''
    # Explicación
    Le damos una imagen a color y lo que hará será reescalarla al tamaño
    que queramos size = [Lx,Ly].
    # Datos de entrada
    name: nombre de la imagen que queremos transformar
    size: tamaño al que queremos convertir la imagen.
    '''
    image = Image.open(name)
    image = image.resize(size=size)                     # Me la cambia de tamaño      
    image = image.convert('1')                          # Me la convierte en 0 y 1
    image_array = np.array(image)*2-1                   # Convierto la imagen en una matriz de -1 y 1
    name2 = name[name.find('/')+1:]
    name2 = 'output_img/'+name2[:name2.find('.')]+'_processed.png'
    
    image.save(name2)

    # Las representamos ya dentro de la función
    fig      = plt.figure()
    original = img.imread(name)
    fig.add_subplot(1,2,1)
    plt.imshow(original,cmap='Grays')
    plt.title('Original')
    plt.axis('off')

    fig.add_subplot(1,2,2)
    plt.matshow(image_array,cmap=plt.cm.gray,fignum=0)
    plt.axis('off')
    plt.title('Procesada')
    plt.show()
    
    return image_array

def noise(image:np.ndarray,p:float):
    shape = image.shape
    image_flatten = image.flatten()
    for i in range(len(image_flatten)):
        if p>rd.random():
            image_flatten[i]=image_flatten[i]*(-1)
        else:
            pass
    image = image_flatten.reshape(shape)
    
    # img = Image.fromarray(image,mode='L')
    # img.save('init_state.png')
    return image

def delete_image(image:np.array,percentage:float)->np.ndarray:
    '''
    # Explicación
    Función que elimina el tanto % de la imagen.
    # Parámetros de entrada
    image: imagen previamente procesada.
    percetage: porcentaje de la imagen que queremos eliminar (de derecha a inzquierda).
    Lo damos en formato decimal.
    # Parámetros de salida
    Nos devuelve la imagen como una matriz con las filas borradas
    '''
    shape = image.shape
    ncols_delete = int(percentage*shape[1])
    image[:,-ncols_delete:]=1
    return image

def random_image(L:int)->np.ndarray:
    '''
    # Explicación
    Genera una imagen aleatoria de blancos y negros (+1,-1) del tamaño que le demos 
    de entrada N=LxL
    # Parámetros de entrada
    L: dimensión de la imagen N=LxL
    # Parámetros de salida
    np.ndarray: Array de dimensión N=LxL
    '''
    N=L**2
    data=np.random.randint(2,size=N)
    data=data*2-1
    data= np.reshape(data,(L,L))
    return data

def change_one(state0:np.ndarray):
    image_shape = state0.shape
    flatten_state = state0.flatten()
    random_neuron = np.random.randint(image_shape[0]**2)
    flatten_state[random_neuron]=flatten_state[random_neuron]*(-1)

    init_state = flatten_state.reshape(image_shape)
    return init_state, random_neuron