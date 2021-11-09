

# --------------------------------EJERCICIO 1 ------------------------------------


# QISKIT


def qiskitrandom():
    import numpy as np
    from qiskit import(
      QuantumCircuit,
      execute,
      Aer)
    from qiskit.visualization import plot_histogram
    
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(1,1)
    circuit.h(0)
    circuit.measure([0],[0])
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print("\nNúmero de ocurrencias para 0 y 1:", counts)



# FOREST


def forestrandom():
    import numpy as np
    from pyquil import Program, get_qc
    from pyquil.gates import *
    from pyquil.api import local_forest_runtime
    
    # construir la puerta Hadamard para el circuito
    p = Program(H(0))
    
    #check make sure of qvm and quilc availability and run the program on a QVM
    with local_forest_runtime():
        qc = get_qc('9q-square-qvm')
        result = qc.run_and_measure(p, trials=1000)
        
    unique, counts = np.unique(result[0], return_counts=True)
    print(unique, counts)






# CIRQ


def cirqrandom():
    import cirq
    
    q = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.H(q), cirq.amplitude_damp(0.2)(q), cirq.measure(q))
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(circuit, repetitions=1000)
    print(result.histogram(key='a'))





# TKET


def tketrandom():
    from pytket import Circuit
    from pytket.backends.ibm import AerBackend
    c = Circuit(1,1) # definir el circuito cin 1 qubit y 1 bit
    c.H(0)           # añadir la Hadamerd al qbit
    c.measure_all()  # medir todos los qbits(1 en este caso)
    
    b = AerBackend()                # conectar al backend
    b.compile_circuit(c)            # compilar el circuito para satisfacer las condiciones del backend
    handle = b.process_circuit(c, n_shots = 1000)  # ejecutar 100 veces
    counts = b.get_result(handle).get_counts()   # recuperar los resultados
    print(counts)





# PROJECTQ


def projectqrandom():
    from projectq import MainEngine  # Se importa el compilador principal
    from projectq.ops import H, Measure  # se importan las operaciones a realizar (Hadamard y Medicion)
    
    eng = MainEngine()  # crear un compildor
    qubit = eng.allocate_qubit()  # se asigna el qbit 
        
    for num in range(100):
        H | qubit  # se aplica la puerta Hadamard
        Measure | qubit  # se mide el qbit resultante
        eng.flush()  # se liberan las puertas y se mide
        print("Measured {}".format(int(qubit)))  # resultado







# --------------------------------EJERCICIO 2 ------------------------------------


def bellstate():
    import numpy as np
    from qiskit import(
      QuantumCircuit,
      execute,
      Aer)
    from qiskit.visualization import plot_histogram, plot_state_city

    # Creamos un objeto Quantum Circuit que actúa sobre el registro cuántico por defecto (q) 
    # de un bit (primer parámetro) y que tiene un registro clásico de un bit (segundo parámetro)
    circuit = QuantumCircuit(2,2)
    # Añadimos una puerta Hadamard con el qubit q_0 como entrada
    circuit.h(0)
    circuit.cnot(0, 1)
    # Mapeamos la medida de los qubits (primer parámetro) sobre los bits clásicos
    circuit.measure([0,1], [0,1])
    # Dibujamos el circuito
    circuit.draw()






#QASM_SIMULATOR


def qasm():
    # Usamos el qasm_simulator de Aer
    simulator_qasm = Aer.get_backend('qasm_simulator')
    # Ejecutamos el circuito sobre el simulador qasm
    job_qasm = execute(circuit, simulator_qasm, shots=1000)
    # Almacenamos los resultados
    result_qasm = job_qasm.result()
    # Capturamos las ocurrencias de salida
    counts_qasm = result_qasm.get_counts(circuit)
    # Escribimos el número de ocurrencias
    print("\nNúmero de ocurrencias:",counts_qasm)
    plot_histogram(counts_qasm)









#STATEVECTOR_SIMULATOR


def statevector():
    simulator_statevector = Aer.get_backend('statevector_simulator')
    # Ejecutamos el circuito sobre el simulador qasm
    job_statevector = execute(circuit, simulator_statevector, shots=1000)
    # Almacenamos los resultados
    result_statevector = job_statevector.result()
    # Capturamos las ocurrencias de salida
    counts_statevector = result_statevector.get_counts(circuit)
    # Escribimos el número de ocurrencias
    print("\nNúmero de ocurrencias:",counts_statevector)
    plot_state_city(result_statevector.get_statevector(circuit), title="Bell initial statevector")









#UNITARY_SIMULATOR


def unitary():
    simulator_unitary = Aer.get_backend('unitary_simulator')

    circ = QuantumCircuit(2,2)
    circ.h(0)
    circ.cx(0, 1)
    
    result = execute(circ, simulator_unitary, shots= 1000).result()
    unitary = result.get_counts(circ)
    print("Circuit unitary:\n", unitary,"\n")
    result_unitary = execute(circ, simulator_unitary, shots=1000).result()
    unitary = result_unitary.get_unitary(circ)
    print("Bell states unitary:\n", unitary)







# --------------------------------EJERCICIO 3 y 4 ------------------------------------


#Estados ghz


def ghzstate():
    import numpy as np
    from qiskit import(
      QuantumCircuit,
      execute,
      Aer)
    from qiskit.visualization import plot_histogram

    # Usamos el qasm_simulator de Aer
    simulator1 = Aer.get_backend('qasm_simulator')
    # Creamos un objeto Quantum Circuit que actúa sobre el registro cuántico por defecto (q) 
    # de un bit (primer parámetro) y que tiene un registro clásico de un bit (segundo parámetro)
    circuit = QuantumCircuit(3, 3)
    # Añadimos una puerta Hadamard con el qubit q_0 como entrada
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    # Mapeamos la medida de los qubits (primer parámetro) sobre los bits clásicos
    circuit.measure([0,1,2], [0,1,2])


    # Ejecutamos el circuito sobre el simulador qasm
    job = execute(circuit, simulator1, shots=1000)
    # Almacenamos los resultados
    result = job.result()
    # Capturamos las ocurrencias de salida
    counts = result.get_counts(circuit)
    # Escribimos el número de ocurrencias
    print("\nNúmero de ocurrencias para 0 y 1:",counts)
    plot_histogram(counts)






#Carga de librerias para ejecucion con ruido y con ordenadores reales



def load():
    from qiskit import QuantumCircuit, execute
    from qiskit import IBMQ, Aer
    from qiskit.visualization import plot_histogram
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.compiler import transpile, assemble, schedule
    IBMQ.save_account('your account', overwrite=True)   
    
    
    
    #No va a funcionar hasta que no pongas tu propia cuenta de IBM
    
    
    
    provider = IBMQ.load_account()




#Ejecucion en simulador con ruido



def noisesimulator():
    # Construir un modelo de ruido a partir de las características de un backend real
    backend = provider.get_backend('ibmq_santiago') # A cambiar para diferentes modelos de ruido
    noise_model = NoiseModel.from_backend(backend)

    # Obtener el mapa de interconexión de los qubits
    coupling_map = backend.configuration().coupling_map

    # Obtener las características de las puertas básicas
    basis_gates = noise_model.basis_gates

    ######################
    # Crear circuito    #
    #####################

    # Perform a noise simulation

    result = execute(circuit, Aer.get_backend('qasm_simulator'),
                     coupling_map=coupling_map,
                     basis_gates=basis_gates,
                     noise_model=noise_model).result()

    ######################
    # Mostrar resultados #
    ######################

    counts = result.get_counts(circuit)
    print("\nIBMQ_SANTIAGO:")
    print("\nNúmero de ocurrencias:",counts)
    plot_histogram(counts)






#Ejecucion en un backend real




def realcomputer():
    ######################
    #   Crear circuito   #
    ######################

    backend = provider.backends.ibmq_belem # A cambiar para diferentes backends
    qobj = assemble(transpile(circuit, backend=backend), backend=backend)
    job = backend.run(qobj)
    retrieved_job = backend.retrieve_job(job.job_id())

    ######################
    # Mostrar resultados #
    ######################

    result = job.result()
    counts = result.get_counts(circuit)
    print("\nIBMQ_BELEM:")
    print("\nNúmero de ocurrencias para 0 y 1:",counts)
    plot_histogram(counts)
