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

total_number_operators = []
for i in range(0, 3):
    n_i = 0
    for s in range(0, 9, 3):
        n_i += of.FermionOperator(
            ((i+s, 1),(i+s, 0))
        )
    total_number_operators.append(n_i)
total_number_operator = of.transforms.jordan_wigner(sum(total_number_operators))

qlcs_total_number_operator = qlcs.observable.create_observable_from_openfermion_text(
    str(total_number_operator)
)

print(qlcs_total_number_operator.get_expectation_value(state))