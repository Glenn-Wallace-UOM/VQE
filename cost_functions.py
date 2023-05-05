import qulacs as qlcs
import openfermion as of

# Normal cost function E
def energy(vqe, theta_list):
    state = qlcs.QuantumState(vqe.n_qubits) # Define all 0 state
    ansatz = vqe.gen_ansatz(theta_list)
    ansatz.update_quantum_state(state)
    cost = vqe.hamiltonian.get_expectation_value(state)
    return (cost, state)

# Cost function which also includes number preservation
def energy_np(vqe, theta_list):
    state = qlcs.QuantumState(vqe.n_qubits) # Define all 0 state
    ansatz = vqe.gen_ansatz(theta_list)
    ansatz.update_quantum_state(state)
    number_operators_spin = []
    for s in range(0, vqe.N):
            n_s = 0
            for i in range(0, vqe.n_qubits, vqe.L):
                n_s += of.FermionOperator(
                    ((i+s, 1),(i+s, 0)))
            number_operators_spin.append(of.transforms.jordan_wigner(n_s))
    
    number_operators_spin = [qlcs.observable.create_observable_from_openfermion_text(
         str(number_operator)).add_operator(1.0, f"I {vqe.n_qubits-1}") for number_operator in number_operators_spin]
    cost = vqe.hamiltonian.get_expectation_value(state) + sum([
         abs(number_operator.get_expectation_value(state)-1) for number_operator in number_operators_spin])
    print(sum([
         number_operator.get_expectation_value(state) for number_operator in number_operators_spin]))
    return (cost, state)