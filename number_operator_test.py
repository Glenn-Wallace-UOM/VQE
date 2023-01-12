import openfermion as of
import numpy as np
import scipy as sp
import qulacs as qlcs
import matplotlib.pyplot as plt
import qulacsvis as qlcsvis
from time import time

from helper import *

state = qlcs.QuantumState(9)

circuit = qlcs.QuantumCircuit(9)

circuit.add_X_gate(0)
circuit.add_X_gate(3)
circuit.add_X_gate(6)

circuit.update_quantum_state(state)

# Create total number operators for each spin
total_number_operators = []
for i in range(0, 9, 3):
    n_i = 0
    for s in range(0, 3):
        n_i += of.FermionOperator(
            ((i+s, 1),(i+s, 0))
        )
    total_number_operators.append(n_i)
print(total_number_operators)

# Jordan-Wigner Transform TNOs
total_number_operators_jw = []
for n in total_number_operators:
    total_number_operators_jw.append(
        of.transforms.jordan_wigner(n)
    )

# Convert JW TNOs to Qulacs
total_number_operators_qlcs = []
for n in total_number_operators_jw:
    total_number_operators_qlcs.append(
        qlcs.observable.create_observable_from_openfermion_text(
            str(n)
        )
    )

# Calculate 'cost' for each TNO independently
number_cost_functions = []
for n in total_number_operators_qlcs:
    number_cost_functions.append((1-n.get_expectation_value(state))**2)

qlcs_total_number_operator = number_cost_functions

print(qlcs_total_number_operator)

#qlcsvis.circuit_drawer(circuit, "mpl")