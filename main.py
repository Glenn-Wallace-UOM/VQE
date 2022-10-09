import openfermion as of
import numpy as np
import scipy as sp
import qulacs as qlcs
import matplotlib.pyplot as plt
import qulacsvis as qlcsvis

## REFACTOR LATER
class VQE():
    # t hopping terms, V coulomb, U onsite, N number of spins, L number of sites
    def __init__(self, N, L, U, V, t, layers, display_ansatz):
        # Calculate number of qubits
        self.n_qubits = L*N
        # Generate hamiltonian
        self.hamiltonian = self.gen_hamiltonian(N, L, U, V, -t)
        # Set layers
        self.layers = layers
        # Set circuit parameter count
        self.param_count = 3+(21*self.layers)
        # Set display ansatz flag
        self.display_ansatz = display_ansatz
    
    def gen_hamiltonian(self, N, L, U, V, t):
        # Generate openfermion hamiltonian hopping and onsite terms
        print("Generating OpenFermion hamiltonian...")
        n_qubits = self.n_qubits
        hopping_terms = []
        # i represents site, s represents spin component
        for s in range(0, n_qubits, N):
            for i in range(0, L):
                if i+1 < L:
                    hopping_terms.append(
                        of.FermionOperator(
                            ((i+s,1), (i+s+1, 0)), coefficient=t
                        )
                    )
                else:
                    hopping_terms.append(
                        of.FermionOperator(
                            ((i+s,1), (i+s+1-L, 0)), coefficient=t
                        )
                    )
        for j in range(0, len(hopping_terms)):
            hopping_terms.append(
                of.hermitian_conjugated(
                    hopping_terms[j]
                )
            )

        onsite_terms = []
        for i in range(0, L):
            for s in range(0, n_qubits, N):
                if i+s+N < n_qubits:
                    onsite_terms.append(
                        of.FermionOperator(
                            ((i+s,1), (i+s,0), (i+s+N, 1), (i+s+N, 0)), coefficient=U
                        )
                    )
                else:
                    onsite_terms.append(
                        of.FermionOperator(
                            ((i+s,1), (i+s,0), (i, 1), (i, 0)), coefficient=U
                        )
                    )
        density_density_terms = []
        total_number_operators = []
        for i in range(0, L):
            n_i = 0
            for s in range(0, n_qubits, N):
                n_i += of.FermionOperator(
                    ((i+s, 1),(i+s, 0))
                )
            total_number_operators.append(n_i)
        for i in range(0, L):
            density_density_terms.append(
                total_number_operators[i]*total_number_operators[i+1 if i < L-1 else 0]*V
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
        theta_list = theta_list.tolist()
        # Define initialiser circuit
        init_circuit = qlcs.QuantumCircuit(self.n_qubits)
        # Add X and RZ gates to init circuit
        for i in range(0, self.n_qubits, 3):
            init_circuit.add_X_gate(i)
            init_circuit.add_RZ_gate(i, theta_list.pop())
        # Define ansatz circuits
        ansatz_circuits = []
        iswap_targets = [(0,1),(3,4),(6,7),(1,2),(4,5),(7,8)]
        for l in range(0, self.layers):
            circuit = qlcs.QuantumCircuit(self.n_qubits)
            # Add iSWAP gates to ansatz
            for target_pair in iswap_targets:
                circuit.add_gate(self.create_iswap_gate(
                    target_pair[0], target_pair[1], theta_list.pop()
                ))
            # Add controlled RZ gates to ansatz
            for i in range(0, 6):
                circuit.add_gate(self.create_crz_gate(i, i+3, theta_list.pop()))
            # Add RZ gates to ansatz
            for i in range(0, self.n_qubits):
                circuit.add_gate(qlcs.gate.RZ(i, theta_list.pop()))
            # Append ansatz to layer list
            ansatz_circuits.append(circuit)
            # Save ansatz if flag is set
            if self.display_ansatz:
                qlcsvis.circuit_drawer(circuit, "mpl")
                self.display_ansatz = False
        return (init_circuit, ansatz_circuits)

    #  Cost function
    def cost(self, theta_list):
        # Define all 0 state
        state = qlcs.QuantumState(self.n_qubits)
        # Generate circuits using theta list
        init_circuit, ansatz_circuits = self.gen_ansatz(theta_list)
        # Pass state through init circuit
        init_circuit.update_quantum_state(state)
        # Pass initialised state through ansatz circuits
        for circuit in ansatz_circuits:
            circuit.update_quantum_state(state)
        return self.hamiltonian.get_expectation_value(state)

    def run(self):
        print("Running VQE...")
        cost_history = []
        init_theta_list = np.random.rand(self.param_count)*2*np.pi
        cost_history.append(self.cost(init_theta_list))
        method = "BFGS"
        options = {"disp":True, "maxiter":200}
        opt = sp.optimize.minimize(self.cost, init_theta_list,
            method=method, options=options,
            #callback=lambda x: cost_history.append(self.cost(x)),
            callback=lambda x: cost_history.append(self.cost(x))
        )
        return cost_history

    def create_iswap_gate(self, target1, target2, angle):
        iswap_gate_matrix = [
            [1,0,0,0],
            [0,np.cos(angle/2), 1j*np.sin(angle/2),0],
            [0,1j*np.sin(angle/2), np.cos(angle/2),0],
            [0,0,0,1]]
        iswap_gate = qlcs.gate.DenseMatrix(
            [target1, target2], iswap_gate_matrix)
        
        return iswap_gate
    
    def create_crz_gate(self, control, target, angle):
        rz_gate = qlcs.gate.RZ(target, angle)
        crz_mat_gate = qlcs.gate.to_matrix_gate(rz_gate)
        crz_mat_gate.add_control_qubit(control, 0)
        return crz_mat_gate

    
if __name__ == "__main__":
    vqe = VQE(L=3, N=3, U=5, V=10, t=1, layers=3, display_ansatz=True)
    results = vqe.run()

    plt.rcParams["font.size"] = 18
    plt.plot(results, color="red", label="VQE")
    plt.xlabel("Iteration")
    plt.ylabel("Energy expectation value")
    plt.legend()
    plt.show()
