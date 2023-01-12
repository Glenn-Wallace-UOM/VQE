import openfermion as of
import numpy as np
import scipy as sp
import qulacs as qlcs
import matplotlib.pyplot as plt
import qulacsvis as qlcsvis
from time import time

from helper import *

## REFACTOR LATER
class VQE():
    # t hopping terms, V coulomb, U onsite, N number of spins, L number of sites
    def __init__(self, shape, U, V, t, layers, maxiter, display_ansatz):
        # Init some instance vars
        self.N = len(shape)
        self.L = len(shape[0])
        # Calculate particle number
        self.Np = 0
        for colour in shape:
            for site in colour:
                self.Np += site
        self.shape = sum(shape, ())
        self.n_qubits = self.L*self.N # Calculate number of qubits
        self.hamiltonian = self.gen_hamiltonian(U, V, -t) # Generate hamiltonian
        self.layers = layers # Set layers
        # Set circuit parameter count
        self.param_count = (3*self.N*self.L - self.N - self.L)*self.layers + self.Np - self.N
        #self.param_count = 9 + 6
        self.display_ansatz = display_ansatz
        # Init iteration counter
        self.iteration_counter = 0
        # Clear/Create output file
        self.f = open("vqe_iterations", "w").close()
        # Set maximum iteration count
        self.maxiter = maxiter
    
    def gen_hamiltonian(self, U, V, t):
        # Generate openfermion hamiltonian hopping and onsite terms
        # print("Generating OpenFermion hamiltonian...")
        n_qubits = self.n_qubits
        hopping_terms = []
        hopping_terms_conj = []
        for colour in range(0, self.n_qubits, self.L):
            for site in range(0, self.L):
                curr_qubit = colour+site
                next_qubit = curr_qubit + 1
                hopping_terms.append(
                    of.FermionOperator(
                        ((curr_qubit, 1),(next_qubit if next_qubit < colour+self.L else colour, 0)), 
                        coefficient=t
                    )
                )
        for term in hopping_terms:
            hopping_terms_conj.append(
                of.hermitian_conjugated(
                    term
                )
            )
        hopping_terms.extend(hopping_terms_conj)
        onsite_terms = []
        for site in range(0, self.L):
            for colour in range(0, self.n_qubits, self.L):
                for other_colour in range(colour+self.L, self.n_qubits, self.L):
                    onsite_terms.append(
                        of.FermionOperator(
                            ((colour+site, 1), (colour+site, 0), (other_colour+site, 1), (other_colour+site, 0)),
                            coefficient=U
                        )
                    )

        density_density_terms = []
        total_number_operators = []
        for i in range(0, self.L):
            n_i = 0
            for s in range(0, n_qubits, self.N):
                n_i += of.FermionOperator(
                    ((i+s, 1),(i+s, 0))
                )
            total_number_operators.append(n_i)
        for i in range(0, self.L):
            density_density_terms.append(
                total_number_operators[i]*total_number_operators[i+1 if i < self.L-1 else 0]*V
            )

        of_hamiltonian = sum(hopping_terms) + sum(onsite_terms) + sum(density_density_terms)

        of_jw_hamiltonian = of.transforms.jordan_wigner(of_hamiltonian)

        # Convert to Qulacs hamiltonian
        qlcs_hamiltonian = qlcs.observable.create_observable_from_openfermion_text(
            str(of_jw_hamiltonian)
        )

        return qlcs_hamiltonian

    # Ansatz circuit, layers -> repetitions
    def gen_ansatz(self, theta_list):
        # Used to index the theta_list
        param_countdown = 0
        # Define initialiser circuit
        circuit = qlcs.QuantumCircuit(self.n_qubits)
        if self.N == 3:
            for i in range(0, self.n_qubits, self.L):
                circuit.add_RY_gate(i, 1.910633)
            for i in range(0, self.n_qubits, self.L):
                h_gate = qlcs.gate.H(i+1)
                h_gate = qlcs.gate.to_matrix_gate(h_gate)
                h_gate.add_control_qubit(i, 1)
                circuit.add_gate(h_gate)
            for i in range(0, self.n_qubits, self.L):
                circuit.add_CNOT_gate(i+1, i+2)
            for i in range(0, self.n_qubits, self.L):
                circuit.add_CNOT_gate(i, i+1)
            for i in range(0, self.n_qubits, self.L):
                circuit.add_X_gate(i)
        elif self.N == 4:
            for i in range(0, self.n_qubits, self.L):
                circuit.add_H_gate(i+2)
                circuit.add_H_gate(i+3)
            for i in range(0, self.n_qubits, self.L):
                X_gate = qlcs.gate.X(i+1)
                X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
                X_mat_gate.add_control_qubit(i+2, 0)
                X_mat_gate.add_control_qubit(i+3, 0)
                circuit.add_gate(X_mat_gate)
            for i in range(0, self.n_qubits, self.L):
                X_gate = qlcs.gate.X(i)
                X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
                X_mat_gate.add_control_qubit(i+2, 1)
                X_mat_gate.add_control_qubit(i+3, 1)
                circuit.add_gate(X_mat_gate)
            for i in range(0, self.n_qubits, self.L):
                X_gate = qlcs.gate.X(i+2)
                X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
                X_mat_gate.add_control_qubit(i, 1)
                circuit.add_gate(X_mat_gate)
                X_gate = qlcs.gate.X(i+3)
                X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
                X_mat_gate.add_control_qubit(i, 1)
                circuit.add_gate(X_mat_gate)
        # for i in range(0, self.n_qubits):
        #     if self.shape[i] == 1:
        #         circuit.add_X_gate(i)
        #         circuit.add_RZ_gate(i, theta_list[param_countdown])
        #         param_countdown += 1
        # Define ansatz circuit
        for l in range(0, self.layers):
            # Add adjacent iSWAP gates to ansatz
            for site in range(0, self.L-1):
                for colour in range(0, self.n_qubits, self.L):
                    curr_qubit = site+colour
                    next_qubit = curr_qubit + 1
                    circuit.add_gate(create_n_iswap_gate(
                            [curr_qubit, next_qubit], theta_list[param_countdown]
                        ))
                    param_countdown += 1
            for i in range(0, self.n_qubits-self.N):
                # circuit.add_gate(create_crz_gate(i, i+self.N, theta_list[param_countdown]))
                circuit.add_gate(create_qiskit_crz_gate(i, i+self.N, theta_list[param_countdown]))
                param_countdown += 1
            # Add RZ gates to ansatz
            for i in range(0, self.n_qubits):
                # circuit.add_gate(qlcs.gate.RZ(i, theta_list[param_countdown]))
                circuit.add_gate(create_qiskit_rz_gate(i, theta_list[param_countdown]))
                param_countdown += 1
            if self.display_ansatz:
                # print("Ansatz circuit:")
                qlcsvis.circuit_drawer(circuit, "mpl")
                self.display_ansatz = False
        # print(self.param_count-param_countdown)
        return circuit

    #  Cost function
    def cost(self, theta_list):
        state = qlcs.QuantumState(self.n_qubits) # Define all 0 state
        ansatz = self.gen_ansatz(theta_list)
        ansatz.update_quantum_state(state)
        return self.hamiltonian.get_expectation_value(state)

    def param_to_state(self, theta_list):
        state = qlcs.QuantumState(self.n_qubits)
        ansatz = self.gen_ansatz(theta_list)
        ansatz.update_quantum_state(state)
        return state

    def run(self):
        # print("Running VQE...")
        self.cost_history = []
        init_theta_list = np.random.rand(self.param_count)*2*np.pi
        self.cost_history.append(self.cost(init_theta_list))
        method = "BFGS"
        options = {"disp":True, "maxiter":self.maxiter}
        # Define optimiser callback function
        def optimiser_callback(x):
            cost_value = self.cost(x)
            self.cost_history.append((x, cost_value))
            self.iteration_counter += 1
            #print("Iteration", str(self.iteration_counter)+":", cost_value)
            with open("vqe_iterations", "a") as f:
                f.write("Iteration "+str(self.iteration_counter)+": "+str(cost_value))
                f.write("\n")
        # Run optimiser
        opt = sp.optimize.minimize(self.cost, init_theta_list,
            method=method, options=options,
            callback=lambda x: optimiser_callback(x)
        )
        return self.cost_history

if __name__ == "__main__":
    # run_time = time()
    # #shape = ((1, 0, 0),(1, 0, 0),(1, 0, 0))
    shape = list((1,0,0,0) for i in range(4))
    # vqe = VQE(shape=shape, U=5, V=0, t=1, layers=11, maxiter=100000, display_ansatz=False)
    # results = vqe.run()
    # run_time = time() - run_time
    # print("Run time:", run_time, "seconds")

    # best_result = vqe.cost(results[len(results)-1][0])
    # best_params = results[len(results)-1][0]

    # np.save("WSTATEN4L11.npy", best_params)
    # best = np.load("WSTATEN4L11.npy")

    # print(best)
    # print(vqe.cost(best))

    # shape = list((1,0,0,0) for i in range(4))
    # for i in range(1, 11):
    #     vqe = VQE(shape=shape, U=5, V=0, t=1, layers=i, maxiter=100000, display_ansatz=False)

    #     approx = vqe.param_to_state(np.load("./parameters/WSTATEN4L{}.npy".format(i))).get_vector()
    #     exact = np.load("ground_state.npy")
    #     print(np.abs(np.vdot(approx, exact))**2)

    exact_state_vec = np.load("ground_state.npy")
    exact_state = qlcs.QuantumState(16)
    exact_state.load(exact_state_vec)
    exact_reduced_density_mat = qlcs.state.partial_trace(exact_state, [0, 1, 2, 3, 4, 5, 6, 7])

    eigenvals = np.linalg.eigvals(exact_reduced_density_mat.get_matrix())
    exact_entanglement_entropy = 0
    for eig in eigenvals:
        if eig != 0+0j:
            ylogy = eig*np.log2(eig)
            exact_entanglement_entropy -= ylogy
    entanglement_entropy = np.abs(exact_entanglement_entropy)
    print(np.abs(entanglement_entropy))
    print(" ")
    # print(qskt.quantum_info.entropy(qskt.quantum_info.partial_trace(qskt.quantum_info.Statevector(exact_state_vec), [0, 1, 2, 3, 4, 5, 6, 7])))

    for i in range(1, 11):
        vqe = VQE(shape=shape, U=5, V=0, t=1, layers=i, maxiter=100000, display_ansatz=False)
        approx_state = vqe.param_to_state(np.load("./parameters/WSTATEN4L{}.npy".format(i)))
        approx_reduced_density_mat = qlcs.state.partial_trace(approx_state, [0, 1, 2, 3, 4, 5, 6, 7])
        approx_eigenvals = np.linalg.eigvals(approx_reduced_density_mat.get_matrix())
        approx_entanglement_entropy = 0
        for eig in approx_eigenvals:
            if eig != 0+0j:
                ylogy = eig*np.log2(eig)
                approx_entanglement_entropy -= ylogy
        approx_entanglement_entropy = np.abs(approx_entanglement_entropy)
        print(approx_entanglement_entropy)
