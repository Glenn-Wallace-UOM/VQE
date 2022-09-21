import openfermion as of
import qulacs as qlcs
from qulacsvis import circuit_drawer

class VQE():
    # J tunneling, U coloumb, N number of sites
    def __init__(self, N, U, J):
        # Calculate number of qubits
        n_qubits = 3*N # Each site has 3 spins

        # Generate openfermion hamiltonian hopping and onsite terms
        print("Generating OpenFermion hamiltonian...")
        hopping_terms = [
            op + of.hermitian_conjugated(op) for op in (
                of.FermionOperator(
                    ((i, 1), (i + 2, 0)), coefficient=J
                ) for i in range(n_qubits - 2)
            )
        ]
        onsite_terms = [
            of.FermionOperator(
                ((i, 1), (i, 0), (i + 1, 1), (i + 1, 0)), coefficient=U
            ) for i in range(0, n_qubits, 2)
        ]
        of_hamiltonian = sum(hopping_terms) + sum(onsite_terms)

        # Apply jordan wigner transformation
        print("Applying Jordan Wigner transformation...")
        of_jw_hamiltonian = of.transforms.jordan_wigner(of_hamiltonian)

        # Convert to Qulacs hamiltonian
        print("Convering to Qulacs hamiltonian...")
        qlcs_hamiltonian = qlcs.observable.create_observable_from_openfermion_text(
            str(of_jw_hamiltonian)
        )

        # Constructing an ansatz (1 layer)
        circuit = qlcs.QuantumCircuit(n_qubits)
        circuit.add_H_gate(1)
        circuit.add_H_gate(1)
        circuit.add_H_gate(2)
        circuit.add_CNOT_gate(0, 3)

        # Create quantum state
        state = qlcs.QuantumState(n_qubits)
        state.set_zero_state() # Setting state entirely to 0

        circuit.update_quantum_state(state)
        circuit_drawer(circuit, "mpl")

    def cost(theta_list):
        state = qlcs.QuantumState()

vqe = VQE(3, 1, 1)