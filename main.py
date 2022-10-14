import openfermion as of
import numpy as np
import scipy as sp
import qulacs as qlcs
import matplotlib.pyplot as plt
import qulacsvis as qlcsvis

## REFACTOR LATER
class VQE():
    # t hopping terms, V coulomb, U onsite, N number of spins, L number of sites
    def __init__(self, shape, U, V, t, layers, display_ansatz):
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
        self.param_count = self.param_count = (3*self.N*self.L - self.N - self.L)*self.layers + self.Np
        # Set display ansatz flag
        self.display_ansatz = display_ansatz
        # Init iteration counter
        self.iteration_counter = 0
        # Clear/Create output file
        self.f = open("test", "w").close()
    
    def gen_hamiltonian(self, U, V, t):
        # Generate openfermion hamiltonian hopping and onsite terms
        print("Generating OpenFermion hamiltonian...")
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
        print("Hopping Terms:")
        print(hopping_terms)
        print("Onsite Terms:")
        print(onsite_terms)
        print("Density-Density Terms:")
        print(density_density_terms)
        
        # Apply jordan wigner transformation
        print("Applying Jordan Wigner transformation...")
        of_jw_hamiltonian = of.transforms.jordan_wigner(of_hamiltonian)

        # Convert to Qulacs hamiltonian
        print("Converting to Qulacs hamiltonian...")
        qlcs_hamiltonian = qlcs.observable.create_observable_from_openfermion_text(
            str(of_jw_hamiltonian)
        )

        return qlcs_hamiltonian

    # Ansatz circuit, layers -> repetitions
    def gen_ansatz(self, theta_list):
        # Used to index the theta_list
        param_countdown = self.param_count-1
        # Define initialiser circuit
        circuit = qlcs.QuantumCircuit(self.n_qubits)
        for i in range(0, self.n_qubits):
            if self.shape[i] == 1:
                circuit.add_X_gate(i)
                circuit.add_RZ_gate(i, theta_list[param_countdown])
                param_countdown -= 1
        # Calculate iswap_targets
        iswap_targets = []
        for site in range(0, self.L-1):
            for colour in range(0, self.n_qubits, self.N):
                iswap_targets.append((site+colour, site+colour+1))
        # Define ansatz circuit
        for l in range(0, self.layers):
            # Add iSWAP gates to ansatz
            for site in range(0, self.L-1):
                for colour in range(0, self.n_qubits, self.L):
                    curr_qubit = site+colour
                    next_qubit = curr_qubit + 1
                    circuit.add_gate(self.create_iswap_gate(
                            curr_qubit, next_qubit, theta_list[param_countdown]
                        ))
                    param_countdown -= 1
            # Add controlled RZ gates to ansatz
            for i in range(0, self.n_qubits-self.N):
                circuit.add_gate(self.create_crz_gate(i, i+self.N, theta_list[param_countdown]))
                param_countdown -= 1
            # Add RZ gates to ansatz
            for i in range(0, self.n_qubits):
                circuit.add_gate(qlcs.gate.RZ(i, theta_list[param_countdown]))
                param_countdown -= 1
            # Save ansatz if flag is set
            if self.display_ansatz:
                print("Ansatz circuit:")
                qlcsvis.circuit_drawer(circuit, "mpl")
                self.display_ansatz = False
        return circuit

    #  Cost function
    def cost(self, theta_list):
        state = qlcs.QuantumState(self.n_qubits) # Define all 0 state
        circuit = self.gen_ansatz(theta_list) # Generate circuits using theta list
        circuit.update_quantum_state(state) # Pass state through ansatz circuits
        return self.hamiltonian.get_expectation_value(state)

    def run(self):
        print("Running VQE...")
        self.cost_history = []
        init_theta_list = np.random.rand(self.param_count)*2*np.pi
        self.cost_history.append(self.cost(init_theta_list))
        method = "BFGS"
        options = {"disp":True, "maxiter":1000}
        # Define optimiser callback function
        def optimiser_callback(x):
            cost_value = self.cost(x)
            self.cost_history.append(cost_value)
            self.iteration_counter += 1
            #print("Iteration", str(self.iteration_counter)+":", cost_value)
            with open("test", "a") as f:
                f.write("Iteration "+str(self.iteration_counter)+": "+str(cost_value))
                f.write("\n")
        # Run optimiser
        opt = sp.optimize.minimize(self.cost, init_theta_list,
            method=method, options=options,
            callback=lambda x: optimiser_callback(x)
        )
        return self.cost_history

    def create_iswap_gate(self, target1, target2, angle):
        iswap_gate_matrix = [
            [1,0,0,0],
            [0,np.cos(angle/2), -1j*np.sin(angle/2),0],
            [0,-1j*np.sin(angle/2), np.cos(angle/2),0],
            [0,0,0,1]]
        iswap_gate = qlcs.gate.DenseMatrix(
            [target1, target2], iswap_gate_matrix)
        return iswap_gate
    
    def create_crz_gate(self, control, target, angle):
        rz_gate = qlcs.gate.ParametricRZ(target, angle)
        crz_mat_gate = qlcs.gate.to_matrix_gate(rz_gate)
        crz_mat_gate.add_control_qubit(control, 0)
        return crz_mat_gate
    
if __name__ == "__main__":
    shape = ((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0))
    #shape = ((1, 0, 0),(0, 1, 0),(0, 0, 1))
    vqe = VQE(shape=shape, U=1, V=0, t=1, layers=3, display_ansatz=True)
    results = vqe.run()
