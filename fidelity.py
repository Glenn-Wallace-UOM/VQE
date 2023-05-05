from vqe import VQE
import ansatze
import initial_states as initial
import cost_functions as cost
import numpy as np
import scipy as sp
import qulacs as qlcs
from functools import partial
import qiskit
from tqdm import tqdm
from time import time

if __name__ == "__main__":
    # Parameters
    U = 5 # Onsite
    V = 0 # Density-density
    T = 1 # Hopping
    N = 4 # Spins
    L = N # Lattice Sites
    layer_range = range(1, 11) # Range of layers
    maxiter = 100000

    # Assemble VQE
    ansatz = ansatze.default_plus_iswap_beta
    # initial_state = initial.product_state(N, 1)
    initial_state = initial.product_state(N, 1)
    cost_function = cost.energy

    # Calulate Fidelity
    print("Fidelity")
    for i in range(1, 11):
        vqe = VQE(N, L, U, V, T, 
                  i, maxiter, initial_state, 
                  ansatz, cost_function, False)
        approx = vqe.param_to_state(np.load("./out/N{}U{}V{}L{}{}{}.npy".format(
            N, U, V, i,initial_state.LABEL, ansatz.LABEL))).get_vector()
        exact = np.load("ground_state.npy")
        print(np.abs(np.vdot(approx, exact))**2)
    
    # Calculate Energies
    print("Energy")
    for i in range(1, 11):
        vqe = VQE(N, L, U, V, T, 
                  i, maxiter, initial_state, 
                  ansatz, cost_function, False)
        energy = cost_function(vqe, np.load("./out/N{}U{}V{}L{}{}{}.npy".format(
            N, U, V, i,initial_state.LABEL, ansatz.LABEL)))[0]
        print("({},".format(i)+str(energy)+")")