import openfermion as of
import numpy as np
import scipy as sp
import qulacs as qlcs
import matplotlib.pyplot as plt
import qulacsvis as qlcsvis

circuit = qlcs.ParametricQuantumCircuit(3)
gate = qlcs.gate.ParametricPauliRotation(
    [0, 1], 
    [1, 1], 
0)
circuit.add_parametric_gate(gate, 0)
qlcsvis.circuit_drawer(circuit, "mpl")
print(circuit.get_parameter_count())
print(qlcs.gate.ParametricPauliRotation([0], [3], 90))