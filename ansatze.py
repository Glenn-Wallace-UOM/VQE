import qulacs as qlcs
import qulacsvis as qlcsvis
from helper import *

class default:
    LABEL = "DEFAULT"
    def __init__(self, N, L, layers) -> None:
        self.parameters = (3*N*L - N - L)*layers + L
        self.L, self.N, self.layers = N, L, layers
        self.n_qubits = N*L
    def generate(self, vqe, initialise_state, theta_list):
        # Used to index the theta_list
        param_countdown = 0
        # Define initialiser circuit
        circuit = qlcs.QuantumCircuit(self.n_qubits)
        circuit, param_countdown = initialise_state(circuit)
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
                circuit.add_gate(create_qiskit_crz_gate(i, i+self.N, theta_list[param_countdown]))
                param_countdown += 1
            # Add RZ gates to ansatz
            for i in range(0, self.n_qubits):
                circuit.add_gate(create_qiskit_rz_gate(i, theta_list[param_countdown]))
                param_countdown += 1
            if vqe.display_ansatz:
                qlcsvis.circuit_drawer(circuit, "mpl")
                vqe.display_ansatz = False
        return circuit

class default_plus_iswap_beta:
    LABEL = "DEFAULT_PLUS_ISWAP_BETA"
    def __init__(self, N, L, layers) -> None:
        self.parameters = (3*N*L - N - L + (L-1)*N)*layers + L
        self.L, self.N, self.layers = N, L, layers
        self.n_qubits = N*L
    def generate(self, vqe, initialise_state, theta_list):
        # Used to index the theta_list
        param_countdown = 0
        # Define initialiser circuit
        circuit = qlcs.QuantumCircuit(self.n_qubits)
        circuit, param_countdown = initialise_state(circuit)
        # Define ansatz circuit
        for l in range(0, self.layers):
            # Add adjacent iSWAP gates to ansatz
            for site in range(0, self.L-1):
                for colour in range(0, self.n_qubits, self.L):
                    curr_qubit = site+colour
                    next_qubit = curr_qubit + 1
                    circuit.add_gate(create_n_iswap_extra_gate(
                            [curr_qubit, next_qubit], 
                            theta_list[param_countdown],
                            theta_list[param_countdown+1]
                        ))
                    param_countdown += 2
            for i in range(0, self.n_qubits-self.N):
                circuit.add_gate(create_qiskit_crz_gate(i, i+self.N, theta_list[param_countdown]))
                param_countdown += 1
            # Add RZ gates to ansatz
            for i in range(0, self.n_qubits):
                circuit.add_gate(create_qiskit_rz_gate(i, theta_list[param_countdown]))
                param_countdown += 1
            if vqe.display_ansatz:
                qlcsvis.circuit_drawer(circuit, "mpl")
                vqe.display_ansatz = False
        return circuit

class default_nnp:
    LABEL = "NNP"
    def __init__(self, N, L, layers) -> None:
        self.parameters = (3*N*L - N)*layers + L
        self.L, self.N, self.layers = N, L, layers
        self.n_qubits = N*L
    def generate(self, vqe, initialise_state, theta_list):
        # Used to index the theta_list
        param_countdown = 0
        # Define initialiser circuit
        circuit = qlcs.QuantumCircuit(self.n_qubits)
        circuit, param_countdown = initialise_state(circuit)
        # Define ansatz circuit
        for l in range(0, self.layers):
            # Add adjacent iSWAP gates to ansatz
            for site in range(0, self.L):
                for colour in range(0, self.n_qubits, self.L):
                    curr_qubit = site+colour
                    next_qubit = curr_qubit + 1 if curr_qubit < self.n_qubits-1 else 0
                    circuit.add_gate(create_n_iswap_gate(
                            [curr_qubit, next_qubit], theta_list[param_countdown]
                        ))
                    param_countdown += 1
            for i in range(0, self.n_qubits-self.N):
                circuit.add_gate(create_qiskit_crz_gate(i, i+self.N, theta_list[param_countdown]))
                param_countdown += 1
            # Add RZ gates to ansatz
            for i in range(0, self.n_qubits):
                circuit.add_gate(create_qiskit_rz_gate(i, theta_list[param_countdown]))
                param_countdown += 1
            if vqe.display_ansatz:
                qlcsvis.circuit_drawer(circuit, "mpl")
                vqe.display_ansatz = False
        return circuit