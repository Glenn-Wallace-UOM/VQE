import openfermion as of
import numpy as np
import scipy as sp
import qulacs as qlcs
import matplotlib.pyplot as plt
import qulacsvis as qlcsvis

## REFACTOR LATER
class VQE():
    # t hopping terms, V coulomb, U onsite, N number of spins, L number of sites
    def __init__(self, N, L, U, V, t, layers, display_circuit):
        # Calculate number of qubits
        self.n_qubits = L*N
        # Generate hamiltonian
        self.hamiltonian = self.gen_hamiltonian(N, L, U, V, -t)
        # Set layers
        self.layers = layers
        # Set display circuit flag
        self.display_circuit = display_circuit
    
    def gen_hamiltonian(self, N, L, U, V, t):
        # Generate openfermion hamiltonian hopping and onsite terms
        print("Generating OpenFermion hamiltonian...")
        hopping_terms = []
        for i in range(0, self.n_qubits, N):
            for s in range(0, N):
                if i < self.n_qubits-N:
                    hopping_terms.append(
                        of.FermionOperator(((i+s, 1),(i+s+N, 0)),
                            coefficient=t
                        )
                    )
                else:
                    hopping_terms.append(
                        of.FermionOperator(((i+s, 1),(s, 0)),
                            coefficient=t
                        )
                    )
        for i in range(0, len(hopping_terms)):
            hopping_terms.append(
                of.hermitian_conjugated(
                    hopping_terms[i]
                )
            )

        onsite_terms = []
        for i in range(0, self.n_qubits, N):
            for s in range(0, N):
                for s_prime in range(s+1, N):
                    onsite_terms.append(
                        of.FermionOperator(((i+s,1),(i+s,0),(i+s_prime,1),(i+s_prime,0)),
                            coefficient=U
                        )
                    )

        of_hamiltonian = sum(hopping_terms) + sum(onsite_terms)
        print("Hopping Terms:")
        print(hopping_terms)
        print("Onsite Terms:")
        print(onsite_terms)

        # Apply jordan wigner transformation
        print("Applying Jordan Wigner transformation...")
        of_jw_hamiltonian = of.transforms.jordan_wigner(of_hamiltonian)

        # Convert to Qulacs hamiltonian
        print("Converting to Qulacs hamiltonian...")
        qlcs_hamiltonian = qlcs.observable.create_observable_from_openfermion_text(
            str(of_jw_hamiltonian)
        )

        print(qlcs_hamiltonian.get_qubit_count())

        return qlcs_hamiltonian

    # Ansatz circuit, layers -> repetitions
    def gen_ansatz_circuit(self, theta_list):
        # Define ansatz circuit and add H gate
        circuit = qlcs.QuantumCircuit(self.n_qubits)      
        # Add X gates
        for i in range(0, self.n_qubits, 3):
            circuit.add_X_gate(i)

        params = 0

        # Add RZ gates
        for i in range(0, self.n_qubits//3):
            circuit.add_gate(qlcs.gate.RZ(i*3, theta_list[i]))
            params += 1

        iswap_target1 = [0,3,6,1,4,7]
        iswap_target2 = [1,4,7,2,5,8]

        for l in range(1, self.layers+1):        
            for i in range(0, len(iswap_target1)):
                circuit.add_gate(self.create_iswap_gate(
                    iswap_target1[i], iswap_target2[i], theta_list[(i*l)+3]
                ))
                params += 1
            # Add controlled rz gates
            for i in range(0, 6):
                circuit.add_gate(self.create_crz_gate(i, i+3, theta_list[(i*l)+9]))
                params += 1
            # Add rz gates
            for i in range(0, self.n_qubits):
                circuit.add_gate(qlcs.gate.RZ(i, theta_list[(i*l)+15]))
                params += 1

        if (self.display_circuit):
            qlcsvis.circuit_drawer(circuit, "mpl")
            self.display_circuit = False
        return circuit

    #  Cost function
    def cost(self, theta_list):
        state = qlcs.QuantumState(self.n_qubits)
        circuit = self.gen_ansatz_circuit(theta_list)
        circuit.update_quantum_state(state)
        # print("Probability of qubit being 1:")
        # print([1-round(state.get_zero_probability(i), 4) for i in range(0, self.n_qubits)])
        return self.hamiltonian.get_expectation_value(state)

    def run(self):
        cost_history = []
        init_theta_list = np.random.rand(3+21*self.layers)*2*np.pi
        cost_history.append(self.cost(init_theta_list))
        print(cost_history)
        method = "BFGS"
        options = {"disp":True, "maxiter":1000}
        opt = sp.optimize.minimize(self.cost, init_theta_list,
            method=method, options=options,
            callback=lambda x: cost_history.append(self.cost(x)),
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
    vqe = VQE(L=3, N=3, U=5, V=0, t=1, layers=1, display_circuit=True)
    results = vqe.run()

    plt.rcParams["font.size"] = 18
    plt.plot(results, color="red", label="VQE")
    plt.xlabel("Iteration")
    plt.ylabel("Energy expectation value")
    plt.legend()
    plt.show()
