import numpy as np
import qulacs as qlcs

def create_cx_gate(target, control):
    X_gate = qlcs.gate.X(target)
    X_mat_gate = qlcs.gate.to_matrix_gate(X_gate)
    X_mat_gate.add_control_qubit(control, 1)
    return X_mat_gate

def create_cry_gate(target, control, angle):
    RY_gate = qlcs.gate.RY(target, angle)
    RY_mat_gate = qlcs.gate.to_matrix_gate(RY_gate)
    RY_mat_gate.add_control_qubit(control, 1)
    return RY_mat_gate

def create_iswap_gate(target1, target2, angle):
    half_angle = angle/2
    sin_ha = np.sin(half_angle)
    cos_ha = np.cos(half_angle)
    iswap_gate_matrix = [
        [1,0,0,0],
        [0,cos_ha, -1j*sin_ha,0],
        [0,-1j*sin_ha, cos_ha,0],
        [0,0,0,1]]
    iswap_gate = qlcs.gate.DenseMatrix(
        [target1, target2], iswap_gate_matrix)
    return iswap_gate

def create_iswap_extra_gate(target1, target2, angle, beta):
    half_angle = angle/2
    sin_ha = np.sin(half_angle)
    cos_ha = np.cos(half_angle)
    exp_b = np.exp(1j*beta)
    iswap_gate_matrix = [
        [1,0,0,0],
        [0,cos_ha, -1j*sin_ha*exp_b,0],
        [0,-1j*sin_ha*exp_b, cos_ha,0],
        [0,0,0,1]]
    iswap_gate = qlcs.gate.DenseMatrix(
        [target1, target2], iswap_gate_matrix)
    return iswap_gate

def add_param_iswap_gate(circuit, target1, target2):
    circuit.add_RZ_gate(target1, -np.pi/2)
    circuit.add_RZ_gate(target2, np.pi/2)
    circuit.add_sqrtX_gate(target1)
    circuit.add_RZ_gate(target1, np.pi/2)
    circuit.add_CNOT_gate(target1, target2)
    circuit.add_sqrtX_gate(target1)
    circuit.add_sqrtX_gate(target2)
    circuit.add_parametric_RZ_gate(target1, np.pi - 0.5*5)
    circuit.add_parametric_RZ_gate(target2, np.pi - 0.5*5)
    circuit.add_sqrtX_gate(target1)
    circuit.add_sqrtX_gate(target2)
    circuit.add_RZ_gate(target1, 7*np.pi/2)
    circuit.add_RZ_gate(target2, 3*np.pi)
    circuit.add_CNOT_gate(target1, target2)
    circuit.add_sqrtX_gate(target1)
    circuit.add_RZ_gate(target2, -np.pi/2)
    circuit.add_RZ_gate(target1, -np.pi/2)

def add_param_crz_gate(circuit, target, control):
    circuit.add_parametric_RZ_gate(control, np.pi/2)
    circuit.add_CNOT_gate(target, control)
    circuit.add_parametric_RZ_gate(control, -np.pi/2)
    circuit.add_CNOT_gate(target, control)

def create_n_iswap_gate(targets, angle):
    n = len(targets)
    half_angle = angle/2
    sin_ha = np.sin(half_angle)
    cos_ha = np.cos(half_angle)
    iswap_gate_matrix = [
        [1,0,0,0],
        [0,cos_ha, -1j*sin_ha,0],
        [0,-1j*sin_ha, cos_ha,0],
        [0,0,0,1]]
    n_iswap_gate = []
    n_iswap_gate = qlcs.gate.merge([
        qlcs.gate.DenseMatrix([targets[i], targets[i+1]], iswap_gate_matrix)
        for i in range(0, n-1)
    ])
    return n_iswap_gate

def create_n_iswap_extra_gate(targets, angle, beta):
    n = len(targets)
    half_angle = angle/2
    sin_ha = np.sin(half_angle)
    cos_ha = np.cos(half_angle)
    iswap_gate_matrix = [
        [1,0,0,0],
        [0,cos_ha, -1j*sin_ha*np.exp(1j*beta),0],
        [0,-1j*sin_ha*np.exp(-1j*beta), cos_ha,0],
        [0,0,0,1]]
    n_iswap_gate = []
    n_iswap_gate = qlcs.gate.merge([
        qlcs.gate.DenseMatrix([targets[i], targets[i+1]], iswap_gate_matrix)
        for i in range(0, n-1)
    ])
    return n_iswap_gate

def create_crz_gate(control, target, angle):
    rz_gate = qlcs.gate.ParametricRZ(target, angle)
    crz_mat_gate = qlcs.gate.to_matrix_gate(rz_gate)
    crz_mat_gate.add_control_qubit(control, 0)
    return crz_mat_gate

def create_qiskit_crz_gate(control, target, angle):
    rz_gate = create_qiskit_rz_gate(target, angle)
    crz_gate = qlcs.gate.to_matrix_gate(rz_gate)
    crz_gate.add_control_qubit(control, 1)
    return crz_gate

def create_n_rz_gate(self, targets, angle):
    rz_gates = []
    for t in targets:
        rz_gates.append(qlcs.gate.RZ(t, angle))
    rz_gates = qlcs.gate.merge(rz_gates)
    return rz_gates

def create_qiskit_rz_gate(target, angle):
    rz_gate_matrix = [
        [np.exp(-1j*(angle/2)), 0],
        [0, np.exp(1j*(angle/2))]
    ]
    rz_gate = qlcs.gate.DenseMatrix(
        target, rz_gate_matrix)
    return rz_gate