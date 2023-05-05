from vqe import VQE
import ansatze
import initial_states as initial
import cost_functions as cost
import numpy as np
import qulacs as qlcs

# Calculate entanglement entropy for a given Qulacs state and partitioning
def entanglement_entropy(partition, state):
    reduced_density_matrix = qlcs.state.partial_trace(
        state, partition).get_matrix()
    eigenvals = np.linalg.eigvals(reduced_density_matrix)
    entanglement_entropy = 0
    for eig in eigenvals:
        if eig != 0+0j:
            ylogy = eig*np.log2(eig)
            entanglement_entropy -= ylogy
    entanglement_entropy = np.abs(entanglement_entropy)
    return entanglement_entropy

# Run VQE using parameters for range of layers
def run_vqe(N, L, U, V, T, layer_range, initial_parameters, 
display_ansatz, maxiter, initial_state, ansatz, cost_function):
    for layers in layer_range:
        vqe = VQE(
            N, L, U, V, T, 
            layers, maxiter, initial_state, ansatz, cost_function, 
            initial_parameters[layers], display_ansatz)
        vqe.run()

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
    ansatz = ansatze.default # add _plus_iswap_beta if needed
    display_ansatz = False # Display a diagram of the ansatz?
    initial_state = initial.product_state(N, 1) # Possible values: initial.product_state, initial.w_state
    initial_parameters = [[] for i in layer_range] # Empty means random
    cost_function = cost.energy
    modification = "" # Possible values: _HOP_FIRST, _PLUS_ISWAP_BETA
    partition = [i for i in range(0, int(N*L/2))] # Traced out qubits for entanglement entropy

    # Load previous parameters
    # initial_parameters = [np.load("./out/N{}U{}V{}L{}{}{}{}.npy".format(
    #     N, U, V, layers ,initial_state.LABEL, ansatz.LABEL, modification)) for layers in layer_range]

    #Run VQE using parameters for range of layers
    run_vqe(N, L, U, V, T, layer_range, initial_parameters, 
            display_ansatz, maxiter, initial_state, ansatz, cost_function)

    # Calulate Fidelity
    print("Fidelity")
    for layers in layer_range:
        vqe = VQE(N, L, U, V, T, 
                  layers, maxiter, initial_state, 
                  ansatz, cost_function, initial_parameters[layers-1], False)
        approx = vqe.param_to_state(np.load("./out/N{}U{}V{}L{}{}{}{}.npy".format(
            N, U, V, layers,initial_state.LABEL, ansatz.LABEL, modification))).get_vector()
        exact = np.load("ground_state.npy")
        fidelity = np.abs(np.vdot(approx, exact))**2
        print("({},".format(layers)+str(fidelity)+")")
    
    # Calculate Energies
    print("Energy")
    for layers in layer_range:
        vqe = VQE(N, L, U, V, T, 
                  layers, maxiter, initial_state, 
                  ansatz, cost_function, initial_parameters[layers-1], False)
        energy = cost_function(vqe, np.load("./out/N{}U{}V{}L{}{}{}{}.npy".format(
            N, U, V, layers,initial_state.LABEL, ansatz.LABEL, modification)))[0]
        print("({},".format(layers)+str(energy)+")")

    # Calculate Entanglement Entropy
    print("Entanglement Entropy")
    for layers in layer_range:
        vqe = VQE(N, L, U, V, T, 
                  layers, maxiter, initial_state, 
                  ansatz, cost_function, initial_parameters[layers-1], False)
        state = vqe.param_to_state(np.load("./out/N{}U{}V{}L{}{}{}{}.npy".format(
            N, U, V, layers,initial_state.LABEL, ansatz.LABEL, modification)))
        entanglement = entanglement_entropy(partition, state)
        print("({},".format(layers)+str(entanglement)+")")
        
