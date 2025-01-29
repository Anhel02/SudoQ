# **Quantum Sudoku Solver using Grover's Algorithm**  

This project presents a quantum computing approach to solving a 4x4 Sudoku puzzle using Grover’s algorithm. Leveraging IBM Quantum services via Qiskit, it offers the flexibility to run on both simulators and real quantum computers. 

## **Team**

This project was developed as part of the final assignment for the Quantum Technologies course at Universidad Europea de Valencia.

Developed by:

- [Ángel Gómez González](https://github.com/Anhel02)
- [Alberto Martínez Gallardo](https://github.com/albrtgallardo)
- [Javier López Roda]()

## **Explanation**

A brief theoretical explanation can be found in the slides.

As for the code explanation, it is provided in the docstrings in the `sudoq.py` file.

Essentially, we input the number of qubits for the desired state and the constraints of the Sudoku puzzle to obtain the most probable solution.

## **Dependencies**  

- Python 3.12.x
- qiskit==1.3.2
- qiskit-aer==0.16.0
- qiskit-ibm-runtime==0.34.0
- matplotlib>=3.10

```bash
pip install -r requirements.txt
```

## **Usage**

This program is specifically configured for the Sudoku example provided in the slides. To solve a different 4x4 Sudoku puzzle, the constraints and the number of qubits in the state must be adjusted accordingly.

The simulation can be run using Python with the following command:

```bash
python sudoq.py
```

Upon completion, the program will display the elapsed time, the most probable state with the solution's precision based on the number of shots, and a histogram plot of the top 5 most probable solutions:

```python
Elapsed time: 0.01s
|00011000001011> : 350 shots (0.1335%)
```

The user will then be prompted to initiate a real simulation:

```python
Real Simulation? (0: No, 1: Yes):
```

If the user inputs `0`, the program will terminate. If the user inputs `1`, the program will log into the IBM Quantum Runtime using the token and username specified in the `credentials.json` file. It will then search for the least busy quantum computer available, execute the simulation, and again display the most probable state along with the updated histogram showing the 5 most probable solutions.

## **Real Simulation**

To run the real simulation on IBM Quantum's platform, you need to configure your `credentials.json` file with your IBM Quantum API token and account name.

Example `credentials.json` file:

  ```json
  {
    "token" : "YOUR_IBM_QUANTUM_API_TOKEN"
    "name" : "YOUR_ACCOUNT_NAME"
  }
```
