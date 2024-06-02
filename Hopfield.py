import numpy as np

class HopfieldNet:
    def __init__(self,img_shape:tuple):
        '''
        # Parámetros de entrada
        img_shape: tamaño de la imagen previamente procesada
        '''
        self.img_shape = img_shape
        self.neurons = img_shape[0]*img_shape[1]
        self.weights = np.zeros((self.neurons,self.neurons))

    def train(self,patterns:np.array):
        '''
        # Explicación
        El parámetro de entrada será los patrones que queremos que nuestra red almacene. 
        Se lo damos en forma de matriz por que una imagen no es mas que una matriz.
        Utilizamos la regla de Hebb para el cálculo de los pesos
        # Parámetros de entrada
        patterns: lista con las imagenes en un array de dimensión igual al nº de patrones a almacenar
            y cada uno de los elementos son las imagenes procesadas en un vector de una dimensión
        '''

        # Regla de aprendizaje que viene dada por la regla de Hebb
        for j in patterns:
            j.flatten()
            # Producto tensorial de cada uno de los vectores consigo mismo
            self.weights += np.outer(j,j)
        # No hay conexiones propias
        np.fill_diagonal(self.weights,0)
        # Normalizamos la matriz de pesos dividiendola por el número de vectores dados
        self.weights /= len(patterns)

    def update(self,state0:np.ndarray,steps:int,f_activation)->np.ndarray:
        '''
        # Explicación
        Introducimos un estado inicial parecido (con ruido, media imagen...) a los patrones que 
        ya había almacenado la red.
        Actualizamos el patrón eligiendo una neurona (pixel) al azar y en función
        de la regla de activación que pongamos, seremos más o menos preciosos.
        # Parámetros de entrada
        state0: patrón inicial previamente procesado.
        steps: número de pasos hasta los que iterará nuestra red.
        f_activation: función de activación de nuestra red. 
        # Parámetros de salida
        Nos devolverá el estado evolucionado de la red en el número de pasos establecidos.
        '''
        state = state0.flatten()
        # Aplanamos el estado inicial de la red
        
        for _ in range(steps):
            # En cada paso se actualizan por lo menos N neuronas, puede que no se actualicen todas
            for _ in range(self.neurons):
                # Escogemos una de las neuronas de manera aleatoria para actualizar su estado
                i = np.random.randint(self.neurons)
                # Calculamos a_i=sum(w_ij*x_j)
                a_i = np.dot(self.weights[i,:],state)
                # Actualizamos la neurona según la función de activación seleccionada
                state[i]=f_activation(a_i)
        return state.reshape((self.img_shape[0],self.img_shape[1]))
        
    def update_slow(self,state0:np.ndarray,steps:int,f_activation,n:float)->np.ndarray:
        state = state0.flatten()
        # Aplanamos el estado inicial de la red
        neurons_update = int(n*self.neurons)
        for _ in range(steps):
            # En cada paso se actualizan por lo menos N neuronas, puede que no se actualicen todas
            for _ in range(neurons_update):
                # Escogemos una de las neuronas de manera aleatoria para actualizar su estado
                i = np.random.randint(self.neurons)
                # Calculamos a_i=sum(w_ij*x_j)
                a_i = np.dot(self.weights[i,:],state)
                # Actualizamos la neurona según la función de activación seleccionada
                state[i]=f_activation(a_i)
        return state.reshape((self.img_shape[0],self.img_shape[1]))

    def energy(self, state0:np.ndarray)->float:
        '''
        # Explicación
        Calculamos la energía que tiene la red neuronal usando el paralelismo con el 
        modelo de Ising:
            E=1/2 * sum(w_ij * s_i * s_j) - sum(b_i * s_i)
        En nuestro caso no hay bias ($b_i=0).
        # Parámetros de entrada
        state0: estado inicial de nuestra red
        # Parámetros de salida 
        energy: la energía de nuestra red como un float
        '''
        state_flatten = state0.flatten()
        
        energy = -0.5 * np.dot(np.dot(state_flatten.T,self.weights),state_flatten)

        return energy





