import numpy as np
import matplotlib.pyplot as plt
from Hopfield import HopfieldNet
from process import random_image, change_one
from tqdm import tqdm
import time
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


# Vamos a definir de manera cuantitativa la capacidad de la red.

def measure_capacity(L):
    N = L**2
    k=int(1.0*N)
    RUNS = 1000
    kall = np.arange(1,k,int(0.05*k))
    hopfield = HopfieldNet([L,L])
    overlap_M = []
    # overlap_M2 = []

    for kk in tqdm(kall):
        time.sleep(0.01)

        overlap = []        
        overlap2 = []
        patterns = []
        for _ in range(kk):
            patterns.append(random_image(L))

        for M in range(RUNS):
        
            hopfield.train(patterns)
            init_state  = patterns[np.random.randint(0,len(patterns))]
            
            state, n_state = change_one(init_state)                       
            final_state = hopfield.update(state,100,np.sign)
            overlap_n = (1-(0.5/N)*sum(abs(init_state.flatten()-final_state.flatten())))
            overlap.append(overlap_n)
        
        overlap_M.append(np.mean(overlap))
        
    return kall/N, overlap_M

x1, overlap1 = measure_capacity(10)
x2, overlap2 = measure_capacity(20)
x3, overlap3 = measure_capacity(30)

fig, ax = plt.subplots()
ax.plot(x1, overlap1, label = "$L=10$")
ax.plot(x2, overlap2, label = "$L=20$")
ax.plot(x3, overlap3, label = "$L=30$")
ax.set_xlim(xmin=0,xmax=0.5)
ax.set_xlabel("$\\alpha$")
ax.set_ylabel("$\\varphi$")
ax.legend()

plt.savefig('figura1.svg')
plt.show()
