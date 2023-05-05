from itertools import chain
import qulacs as qlcs

class product_state:
    LABEL = "PROD"
    def __init__(self, size, position) -> None:
        self.size = size
        self.position = position

        shape_tuple = list((0 for i in range(self.size)))
        shape_tuple[self.position-1] = 1
        shape = list((shape_tuple for i in range(self.size)))
        shape = list(chain.from_iterable(shape))

        self.shape = shape
        self.parameters = size
    def generate(self, vqe, theta_list, circuit):
        param_countdown = 0
        try:
            for i in range(0, vqe.n_qubits):
                if self.shape[i] == 1:
                    circuit.add_X_gate(i)
                    circuit.add_RZ_gate(i, theta_list[param_countdown])
                    param_countdown += 1
        except:
            print(theta_list)
        return (circuit, param_countdown)

class w_state:
    LABEL = "W"
    def __init__(self) -> None:
        self.parameters = 0
    def generate(self, vqe, theta_list, circuit):
        param_countdown = 0
        if vqe.N == 3:
            for i in range(0, vqe.n_qubits, vqe.L):
                circuit.add_RY_gate(i, 1.910633)
            for i in range(0, vqe.n_qubits, vqe.L):
                h_gate = qlcs.gate.H(i+1)
                h_gate = qlcs.gate.to_matrix_gate(h_gate)
                h_gate.add_control_qubit(i, 1)
                circuit.add_gate(h_gate)
            for i in range(0, vqe.n_qubits, vqe.L):
                circuit.add_CNOT_gate(i+1, i+2)
            for i in range(0, vqe.n_qubits, vqe.L):
                circuit.add_CNOT_gate(i, i+1)
            for i in range(0, vqe.n_qubits, vqe.L):
                circuit.add_X_gate(i)
        elif vqe.N == 4:
            for i in range(0, vqe.n_qubits, vqe.L):
                circuit.add_H_gate(i+2)
                circuit.add_H_gate(i+3)
            for i in range(0, vqe.n_qubits, vqe.L):
                X_gate = qlcs.gate.X(i+1)
                X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
                X_mat_gate.add_control_qubit(i+2, 0)
                X_mat_gate.add_control_qubit(i+3, 0)
                circuit.add_gate(X_mat_gate)
            for i in range(0, vqe.n_qubits, vqe.L):
                X_gate = qlcs.gate.X(i)
                X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
                X_mat_gate.add_control_qubit(i+2, 1)
                X_mat_gate.add_control_qubit(i+3, 1)
                circuit.add_gate(X_mat_gate)
            for i in range(0, vqe.n_qubits, vqe.L):
                X_gate = qlcs.gate.X(i+2)
                X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
                X_mat_gate.add_control_qubit(i, 1)
                circuit.add_gate(X_mat_gate)
                X_gate = qlcs.gate.X(i+3)
                X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
                X_mat_gate.add_control_qubit(i, 1)
                circuit.add_gate(X_mat_gate)
        return (circuit, param_countdown)

