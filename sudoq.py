import itertools
import time
import operator

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

import matplotlib.pyplot as plt


class IBMRuntime():
    """
    A class to manage IBM Quantum credentials and interact with the 
    Qiskit Runtime Service for quantum computing tasks.

    Attributes:
        credentials (dict): A dictionary containing IBM Quantum credentials 
            loaded from a local JSON file. Expected keys in the JSON file:
            - 'token': The IBM Quantum API token.
            - 'name': The name of the account.
    """

    def __init__(self):
        """
        Initializes the IBMRuntime instance by loading IBM Quantum credentials 
        from a JSON file.
        """

        self.credentials = eval(open('credentials.json', 'r').read())
        
        if self.credentials['token'] == 'YOUR_IBM_QUANTUM_API_TOKEN' or self.credentials['name'] == 'YOUR_ACCOUNT_NAME':
            raise ValueError("Error: Change the credentials on the JSON file.")

    def qiskit_sign(self):
        """
        Signs up for the Qiskit Runtime Service using the loaded credentials 
        and saves the account as the default account.

        Returns:
            QiskitRuntimeService: An instance of the Qiskit Runtime Service 
            authenticated with the saved credentials.
        """

        QiskitRuntimeService.save_account(
            channel ='ibm_quantum',
            token = self.credentials['token'],
            set_as_default=True,
            name = self.credentials['name'],
            overwrite=True
        )
         
        service = QiskitRuntimeService(name=self.credentials['name'])
        return service

    def qiskit_log(self):
        """
        Logs into the Qiskit Runtime Service using the saved credentials.

        Returns:
            QiskitRuntimeService: An instance of the Qiskit Runtime Service 
            authenticated with the loaded credentials.
        """

        service = QiskitRuntimeService(name=self.credentials['name'])
        return service


class SudoQ:
    """
    A class to implement Grover's algorithm to solve a 4x4 Sudoku puzzle 
    using a quantum approach. The class supports both simulation on a QASM 
    simulator and execution on a real quantum computer.

    This class encodes the Sudoku problem into a quantum circuit, utilizes 
    Grover's algorithm to find the solution, and then interprets the results.

    Attributes:
        shots (int): Number of executions of the circuit.
        real_shots (int): Number of executions for the real simulation.
        nQbits (int): Number of qubits in the circuit.
        q (QuantumRegister): Quantum register for the circuit.
        c (ClassicalRegister): Classical register for the circuit.
        qc (QuantumCircuit): The quantum circuit that is created and manipulated.
    """

    def __init__(self, nQbits):
        """
        Initializes a quantum circuit with n qubits, applying a Hadamard gate to each one.

        Args:
            nQbits (int): Number of qubits in the circuit.
        """
        
        self.shots = 2048*128
        self.real_shots = 2048*128
        self.nQbits = nQbits
        self.q = QuantumRegister(nQbits)
        self.c = ClassicalRegister(nQbits)
        self.qc = QuantumCircuit(self.q, self.c)
        self.qc.h(range(nQbits))

    def permutation(self, classicalB):
        """
        Generates all possible permutations of the elements not in the provided list.

        Args:
            classicalB (list): List of integers excluded from the permutations.

        Returns:
            list: List of possible permutations of the remaining elements.
        """

        aux = [0, 1, 2, 3]
        list_out = [i for i in aux if i not in classicalB]

        return list(itertools.permutations(list_out))

    def qbit_to_binary(self, perm):
        """
        Converts a list of permutations into their binary representation.

        Args:
            perm (list): List of permutations.

        Returns:
            list: List of binary strings representing the permutations.
        """

        perm_bin = []
        for item in perm:
            binary_num = ''.join(format(i, "02b") for i in item)
            perm_bin.append(binary_num)
        return perm_bin

    def oracle(self, l_qbits, l_perm_bin):
        """
        Implements a quantum oracle that marks target states based on binary permutations.

        Args:
            l_qbits (list): List of qubits used in the circuit.
            l_perm_bin (list): List of binary permutations marked as solutions.
        """

        for item in l_perm_bin:
            for i, j in enumerate(item):
                if j == '0':
                    self.qc.x(l_qbits[i])
            self.qc.h(l_qbits[0])
            self.qc.mcx(l_qbits[1:], l_qbits[0])
            self.qc.h(l_qbits[0])
            for i, j in enumerate(item):
                if j == '0':
                    self.qc.x(l_qbits[i])

    def reflector(self, l_qbits):
        """
        Implements the reflection (diffusion) operator in Grover's algorithm.

        Args:
            l_qbits (list): List of qubits on which the reflection is applied.
        """

        self.qc.h(l_qbits)
        self.qc.x(l_qbits)
        self.qc.h(l_qbits[0])
        self.qc.mcx(l_qbits[1:len(l_qbits)], l_qbits[0])
        self.qc.h(l_qbits[0])
        self.qc.x(l_qbits)
        self.qc.h(l_qbits)

    def grover(self, l_qbits, l_bits):
        """
        Implements Grover's algorithm to search for solutions in a space marked by an oracle.

        Args:
            l_qbits (list): List of qubits used in the circuit.
            l_bits (list): List of classical bits that define exclusions for permutations.
        """

        n_rep = 1 if len(l_qbits) <= 4 else 2
        perm = self.permutation(l_bits)
        perm_bin = self.qbit_to_binary(perm)
        for _ in range(n_rep):
            self.oracle(l_qbits, perm_bin)
            self.reflector(self.qc.qubits)

    def run_simulation(self):
        """
        Runs the simulation of the quantum circuit using a Qiskit simulator.

        Args:
            shots (int, optional): Number of measurements to perform. Defaults to 2048*128.

        Prints:
            Displays the most frequent state and its percentage.

        Displays:
            Histogram of the most frequent measurements.
        """

        self.qc.measure(self.q, self.c)
        simulator = Aer.get_backend('aer_simulator')
        transpiled_qc = transpile(self.qc, simulator)
        result = simulator.run(transpiled_qc, shots=self.shots).result()
        counts = result.get_counts()
        graph_dict = dict(sorted(counts.items(), key=operator.itemgetter(1))[-5:])
        max_state = max(graph_dict, key=graph_dict.get)
        print(f'|{max_state}> : {graph_dict[max_state]} shots ({graph_dict[max_state]/self.shots*100:.4f}%)')
        plot_histogram(graph_dict)
        plt.show()

    def execute_real_computer(self):
        """
        Runs the circuit on a real quantum computer using IBM Quantum.
        
        Prints:
            Displays the most frequent state and its percentage.

        Displays:
            Histogram of the most frequent measurements.
        """

        print('\nConnecting to IBM Runtime...')
        IBM = IBMRuntime()
        IBM.qiskit_sign()
        service = IBM.qiskit_log()

        print('Searching for the least busy computer...')
        backend = service.least_busy(operational=True, simulator=False)
        transpiled_qc = transpile(self.qc, backend)
        print(f'Running the circuit on {backend}...\n')

        sampler = Sampler(backend)
        job = sampler.run([transpiled_qc],shots=self.real_shots)
        result = job.result()
        counts = result[0].data.c0.get_counts()
        ibm_dict = dict(sorted(counts.items(), key=operator.itemgetter(1))[-20:])
        max_state = max(ibm_dict, key=ibm_dict.get)
        print(f'|{max_state}> : {ibm_dict[max_state]} shots ({ibm_dict[max_state]/self.real_shots*100:.4f}%)')
        plot_histogram(ibm_dict)
        plt.show()


if __name__ == '__main__':
    
    nQbits = 14  # Length of the Solution State
    sudoq = SudoQ(nQbits)

    r = 1
    start_time = time.time()

    for _ in range(r):

        # Row Conditions

        sudoq.grover([sudoq.q[0], sudoq.q[1]], [0, 1, 2])                           # First Row
        sudoq.grover([sudoq.q[2], sudoq.q[3], sudoq.q[4], sudoq.q[5]], [2, 3])      # Second Row
        sudoq.grover([sudoq.q[6], sudoq.q[7], sudoq.q[8], sudoq.q[9]], [2, 3])      # Third Row
        sudoq.grover([sudoq.q[10], sudoq.q[11], sudoq.q[12], sudoq.q[13]], [1, 3])  # Forth Row
        
        # Column Conditions

        sudoq.grover([sudoq.q[0], sudoq.q[1], sudoq.q[6], sudoq.q[7]], [1, 2])      # First Column
        sudoq.grover([sudoq.q[2], sudoq.q[3], sudoq.q[10], sudoq.q[11]], [0, 3])    # Second Column
        sudoq.grover([sudoq.q[4], sudoq.q[5], sudoq.q[8], sudoq.q[9]], [2, 3])      # Third Column
        sudoq.grover([sudoq.q[12], sudoq.q[13]], [1, 2, 3])                         # Forth Column

        # Block Conditions

        sudoq.grover([sudoq.q[0], sudoq.q[1], sudoq.q[2], sudoq.q[3]], [0, 2])      # Top-left Block
        sudoq.grover([sudoq.q[4], sudoq.q[5]], [1, 2, 3])                           # Top-right Block
        sudoq.grover([sudoq.q[6], sudoq.q[7], sudoq.q[10], sudoq.q[11]], [1, 3])    # Bottom-left Block
        sudoq.grover([sudoq.q[8], sudoq.q[9], sudoq.q[12], sudoq.q[13]], [2, 3])    # Bottom-right Block

    elapsed_time_grover = time.time() - start_time
    print(f'\nElapsed time: {elapsed_time_grover:.2f}s\n')

    qc = sudoq.run_simulation()

    while True:
        real = input('\nReal Simulation? (0: No, 1: Yes): ')
        if real == '0':
            break
        elif real == '1':
            sudoq.execute_real_computer()
            break
        else:
            print('Select a valid option.')
