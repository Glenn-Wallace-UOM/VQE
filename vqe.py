import openfermion as of
import numpy as np
import scipy as sp
import qulacs as qlcs
import os
from time import time

from helper import *

class VQE():
    # t hopping terms, V coulomb, U onsite, N number of spins, L number of sites
    def __init__(self, N, L, U, V, t, 
                 layers, maxiter, initial_state, ansatz_class, cost_function, 
                 initial_parameters = [], display_ansatz = False, file_label = ""):
        # Init some instance vars
        ansatz = ansatz_class(N, L, layers)
        self.N, self.L, self.U, self.V = N, L, U, V
        self.initial_parameters = initial_parameters
        self.gen_ansatz = lambda theta_params: ansatz.generate(self,
            lambda circuit: initial_state.generate(self, theta_params, circuit), theta_params)
        self.Np = N*L # Calculate particle number
        self.n_qubits = self.L*self.N # Calculate number of qubits
        self.hamiltonian = self.gen_hamiltonian(U, V, -t) # Generate hamiltonian
        self.layers = layers # Set layers
        self.cost = cost_function
        self.initstate_label = initial_state.LABEL
        self.anstatz_label = ansatz_class.LABEL
        # Set circuit parameter count
        self.param_count = ansatz.parameters + initial_state.parameters
        self.display_ansatz = display_ansatz
        # Init iteration counter for output
        self.iteration_counter = 0
        # Clear/Create output file
        if os.path.exists("out"):
            self.f = open("./out/vqe_iterations", "w").close()
        # Set maximum iteration count
        self.maxiter = maxiter
        self.file_label = file_label
    
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
                        coefficient=t))
        for term in hopping_terms:
            hopping_terms_conj.append(
                of.hermitian_conjugated(
                    term))
        hopping_terms.extend(hopping_terms_conj)
        onsite_terms = []
        for site in range(0, self.L):
            for colour in range(0, self.n_qubits, self.L):
                for other_colour in range(colour+self.L, self.n_qubits, self.L):
                    onsite_terms.append(
                        of.FermionOperator(
                            ((colour+site, 1), (colour+site, 0), (other_colour+site, 1), (other_colour+site, 0)),
                            coefficient=U))
        density_density_terms = []
        total_number_operators = []
        for i in range(0, self.L):
            n_i = 0
            for s in range(0, n_qubits, self.N):
                n_i += of.FermionOperator(
                    ((i+s, 1),(i+s, 0)))
            total_number_operators.append(n_i)
        for i in range(0, self.L):
            density_density_terms.append(
                total_number_operators[i]*total_number_operators[i+1 if i < self.L-1 else 0]*V)

        of_hamiltonian = sum(hopping_terms) + sum(onsite_terms) + sum(density_density_terms)

        of_jw_hamiltonian = of.transforms.jordan_wigner(of_hamiltonian)

        # Convert to Qulacs hamiltonian
        qlcs_hamiltonian = qlcs.observable.create_observable_from_openfermion_text(
            str(of_jw_hamiltonian)
        )

        self.number_operators = []

        return qlcs_hamiltonian
    
    def param_to_state(self, theta_list):
        state = qlcs.QuantumState(self.n_qubits)
        ansatz = self.gen_ansatz(theta_list)
        ansatz.update_quantum_state(state)
        return state
    
    def run(self):
        # print("Running VQE...")
        if not os.path.exists("out"):
            os.mkdir("out")
        run_time = time()
        self.cost_history = []
        if self.initial_parameters == []:
            init_theta_list = np.random.rand(self.param_count)*2*np.pi
        else:
            print("Using initial parameters...")
            init_theta_list = self.initial_parameters
        self.cost_history.append(self.cost(self, init_theta_list))
        method = "BFGS"
        options = {"disp":True, "maxiter":self.maxiter, "gtol":1e-5}
        # Define optimiser callback function
        def optimiser_callback(x):
            cost_value = self.cost(self, x)
            self.cost_history.append((x, cost_value[0]))
            self.iteration_counter += 1
            #print("Iteration", str(self.iteration_counter)+":", cost_value)
            with open("./out/vqe_iterations", "a") as f:
                f.write("Iteration "+str(self.iteration_counter)+": "+str(cost_value[0]))
                f.write("\n")
        # Run optimiser
        opt = sp.optimize.minimize(lambda guess: self.cost(self, guess)[0], init_theta_list,
            method=method, options=options,
            callback=lambda x: optimiser_callback(x)
        )
        # Calculate how long it took to run
        run_time = time() - run_time
        print("Run time:", run_time, "seconds")
        # Get best result and parameters
        best_params = self.cost_history[len(self.cost_history)-1][0]
        # Save best parameters to a file
        np.save(f"./out/N{self.N}U{self.U}V{self.V}L{self.layers}{self.initstate_label}{self.anstatz_label}{self.file_label}.npy", best_params)
        return opt["fun"]