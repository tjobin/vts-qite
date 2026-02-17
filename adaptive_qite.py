import numpy as np
import scipy.linalg
from itertools import product
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace
from qiskit.circuit.library import UnitaryGate

# --- Efficient Math Helpers (Cached & Vectorized) ---

_BASIS_CACHE = {}

def get_pauli_basis(
        num_qubits : int
        ) -> tuple[list[SparsePauliOp], np.ndarray]:
    
    """
    Returns the Pauli basis for num_qubits qubits as a tuple of a list of 4^num_qubits SparsePauliOp's and
    a numpy array of shape (4^num_qubits, 2^num_qubits, 2^num_qubits) containing the 4^num_qubits Pauli
    matrices of dim 2^num_qubits x 2^num_qubits written in the computational basis.
        Args:
            - num_qubits : int, number of qubits
        Returns:
            - pauli_basis : tuple, of size 2. Takes the form, e.g. for num_qubits = 2,

            (
            [
            SparsePauliOp(['II'], coeffs=[1.+0.j]),
            SparsePauliOp(['IX'], coeffs=[1.+0.j]),
            SparsePauliOp(['IY'], coeffs=[1.+0.j]),
              ...
            SparsePauliOp(['ZX'], coeffs=[1.+0.j]),
            SparsePauliOp(['ZY'], coeffs=[1.+0.j]),
            SparsePauliOp(['ZZ'], coeffs=[1.+0.j])
            ],
            array(
            [[[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]],
            [[ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
                [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j]],
              ...
            [[ 0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j],
                [ 0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j],
                [ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j]],
            [[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]]]
                )
            )
    """

    if num_qubits in _BASIS_CACHE:
        return _BASIS_CACHE[num_qubits]
    labels = ['I', 'X', 'Y', 'Z']
    basis_strings = [''.join(p) for p in product(labels, repeat=num_qubits)]
    basis_ops = [SparsePauliOp(s) for s in basis_strings]
    basis_mats = np.array([op.to_matrix() for op in basis_ops])
    _BASIS_CACHE[num_qubits] = (basis_ops, basis_mats)
    pauli_basis = _BASIS_CACHE[num_qubits]
    return pauli_basis

def get_active_and_domain_qubits(
        term : SparsePauliOp, 
        total_qubits : int, 
        radius : int = 1
        ) -> tuple[list[int], list[int]]:
    
    """
    Returns the active and domain qubits for a given Hamiltonian term. Active qubits are those on which the term acts
    non-trivially (X, Y, Z), while domain qubits include the active qubits and those within a specified radius of them.
        Args:
            - term: SparsePauliOp, the Hamiltonian term for which we want to identify the active and domain qubits
            - total_qubits: int, total number of qubits 
            - radius: int, the radius around active qubits to include in the domain
        Returns:
            - active_indices: list of int, the indices of the active qubits (those on which the term acts non-trivially)
            - sorted_domain_set: list of int, the sorted list of indices of the domain qubits (active qubits + those
            within the specified radius)
    """

    p_str = term.paulis[0].to_label()
    active_indices = []

    # Qiskit string is reversed (q_{n-1}...q_0)
    for i, char in enumerate(reversed(p_str)):
        if char != 'I':
            active_indices.append(i)
    if not active_indices: return [], []
    domain_set = set(active_indices)
    for idx in active_indices:
        for r in range(1, radius + 1):
            if idx - r >= 0: domain_set.add(idx - r)
            if idx + r < total_qubits: domain_set.add(idx + r)
    sorted_domain_set = sorted(list(domain_set))
    return active_indices, sorted_domain_set

def solve_vectorized(
        basis_mats: np.ndarray,
        h_local_mat: np.ndarray,
        rdm_data: np.ndarray,
        ) -> np.ndarray:
    
    """
    Solves S * a = b using vectorized tensor contractions. The matrix S and vector b are obtained exactly;
    TODO : obtain S and b using actual expectation values (e.g. AerEstimator)
        Args:
            - basis_mats: numpy array of shape (4^n, 2^n, 2^n) containing the 4^n Pauli matrices of
            dim 2^n x 2^n written in the computational basis
            - h_local_mat: numpy array of shape (2^n, 2^n) representing the local Hamiltonian term in
            the computational basis
            - rdm_data: numpy array of shape (2^n, 2^n) representing the reduced density matrix of the
            current state on the domain qubits
        Returns:
            - a_coeffs: numpy array of shape (4^n,) containing the coefficients of the Pauli basis elements that define
    """

    B = basis_mats
    rho = rdm_data
    
    # S matrix: S_ij = Re[ Tr(rho * B[i] * B[j]) ]
    S_complex = np.einsum('pq,iqr,jrp->ij', rho, B, B, optimize=True)
    S = np.real(S_complex)
    
    # b vector: b_i = -Im[ Tr(rho * [h, B[i]]) ]
    term1 = np.einsum('pq,qr,irp->i', rho, h_local_mat, B, optimize=True)
    term2 = np.einsum('pq,iqr,rp->i', rho, B, h_local_mat, optimize=True)
    b = -(term1 - term2).imag
    
    # Solve (S + reg) * a = b
    S_reg = S + np.eye(len(b)) * 1e-8
    try:
        a_coeffs = scipy.linalg.solve(S_reg, b, assume_a='sym')
    except:
        a_coeffs = np.linalg.lstsq(S_reg, b, rcond=1e-6)[0]
    return a_coeffs

# --- Adaptive Circuit Step ---

def adaptive_qite_step(
        qc: QuantumCircuit,
        H: list[SparsePauliOp],
        current_delta_tau: float,
        domain_radius: int = 1
        ) -> tuple[QuantumCircuit, float]:
     
    """
    Appends QITE gates for one time step to 'qc'. Returns updated circuit and max_a coefficient.
        Args:
            - qc: QuantumCircuit, the current quantum circuit to which we will append the QITE gates for this time step
            - H: list of SparsePauliOp, the Hamiltonian terms for which we will construct the QITE gates
            - current_delta_tau: float, the imaginary time interval ∆tau for this time step
            - domain_radius: int, the radius around active qubits to include in the domain for each Hamiltonian term
        Returns:
            - qc: QuantumCircuit, the updated quantum circuit with the QITE gates for this time step appended
            - max_a_in_sweep: float, the maximum absolute value of the coefficients 'a' obtained across all Hamiltonian
            terms in this sweep, which can be used for adaptive time step logic
    """
     
    num_qubits = qc.num_qubits
    current_psi = Statevector.from_instruction(qc) # Simulator shortcut
    max_a_in_sweep = 0.0
    
    for term in H:
        if abs(term.coeffs[0]) < 1e-10: continue

        # Identify domain
        active_idx, domain_idx = get_active_and_domain_qubits(term, num_qubits, domain_radius)
        if not domain_idx: continue
        domain_size = len(domain_idx)

        # Compute reduced density matrix
        trace_qubits = [q for q in range(num_qubits) if q not in domain_idx]
        rdm = partial_trace(current_psi, trace_qubits)
        rdm_data = rdm.data

        # Construct local hamiltonian
        full_p_str = term.paulis[0].to_label()
        op_list = [np.eye(2, dtype=complex)] * domain_size
        global_to_local = {g: l for l, g in enumerate(domain_idx)}
        
        for g_idx in active_idx:
            char = full_p_str[len(full_p_str) - 1 - g_idx]
            if char == 'X': mat = np.array([[0, 1], [1, 0]], dtype=complex)
            elif char == 'Y': mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
            elif char == 'Z': mat = np.array([[1, 0], [0, -1]], dtype=complex)
            else: mat = np.eye(2, dtype=complex)
            op_list[domain_size - 1 - global_to_local[g_idx]] = mat

        h_local_mat = op_list[0]
        for mat in op_list[1:]: h_local_mat = np.kron(h_local_mat, mat)
        h_local_mat *= term.coeffs[0]

        # Solve linear system
        basis_ops, basis_mats = get_pauli_basis(domain_size)
        
        if rdm_data.shape != h_local_mat.shape: continue 

        a_coeffs = solve_vectorized(basis_mats, h_local_mat, rdm_data)
        
        # Track max 'velocity' for adaptive logic
        current_max = np.max(np.abs(a_coeffs))
        if current_max > max_a_in_sweep:
            max_a_in_sweep = current_max

        # Construct unitary
        A_matrix = np.einsum('i,ijk->jk', a_coeffs, basis_mats)
        U_matrix = scipy.linalg.expm(-1j * current_delta_tau * A_matrix)
        
        # Append gate
        u_gate = UnitaryGate(U_matrix, label=f"QITE")
        qc.append(u_gate, qargs=domain_idx)
        
        # Update tracking state for next Trotter term
        current_psi = current_psi.evolve(u_gate, qargs=domain_idx)

    return qc, max_a_in_sweep

# --- Main Adaptive Loop ---

def run_adaptive_qite(
        H: list[SparsePauliOp],
        initial_state: QuantumCircuit,
        total_time: float,
        max_steps: int = 1000,
        initial_dt: float = 0.1,
        target_rotation: float = 0.1,
        max_dt: float = 0.5,
        min_dt: float = 1e-4,
        domain_radius: int = 1,
        filename: str = 'default_filename'
        ) -> tuple[QuantumCircuit, list[float], list[float], list[float], list[float]]:
    
    """
    Returns the final QuantumCircuit after performing adaptive circuit QITE, along with the history of energies,
    times, time steps, and max_a values for analysis.
        Args:
            - H: list of SparsePauliOp, the Hamiltonian terms for which we will construct the QITE gates
            - initial_state: QuantumCircuit, the initial state preparation circuit
            - total_time: float, the total imaginary time evolution duration we want to achieve
            - max_steps: int, maximum number of iterations to perform (safety cap)
            - initial_dt: float, the initial imaginary time step ∆tau to start with
            - target_rotation: float, the target rotation angle 𝜃_max (in radians) for the adaptive time step logic
            - max_dt: float, the maximum allowed time step ∆tau to prevent instability
            - min_dt: float, the minimum allowed time step ∆tau to determine convergence
            - domain_radius: int, the radius around active qubits to include in the domain for each Hamiltonian term
            - filename: str, the name of the output file to write the history of the simulation (without extension)
        Returns:
            - qc: QuantumCircuit, the final quantum circuit after performing adaptive circuit QITE
            - energies: list of float, the history of energy expectation values at each accepted step
            - times: list of float, the history of cumulative imaginary time at each accepted step
            - dts: list of float, the history of time steps ∆tau used at each accepted step
            - max_as: list of float, the history of maximum 'a' coefficients obtained at each accepted step 
    """

    # Initialize Circuit
    qc = QuantumCircuit(initial_state.num_qubits)
    qc.prepare_state(initial_state)
        
    current_dt = initial_dt
    time_elapsed = 0.0
    
    # Store history
    energies = []
    times = []
    dts = []
    max_as = []
    
    # Initial Energy Check
    psi_init = Statevector.from_instruction(qc)
    psi_init = psi_init / np.linalg.norm(psi_init.data)
    current_energy = psi_init.expectation_value(H).real

    energies.append(current_energy)
    times.append(0.0)
    dts.append(initial_dt)

    if filename is not None:
        fout = open(f'out/{filename}.out','w')
        fout.write("QITE simulation \n")
        fout.write(f"{'Iteration':>10} {'dtau':>10} {'E':>10} {'∆E':>10} {'max_a':>10}\n")
        fout.write(f"{'0':>10} {current_dt:>10.4f} {current_energy:>10.6f} {'-':>10} {'-':>10}\n")

    
    print(f"Starting Adaptive Circuit QITE. Target Rot={target_rotation} rad")
    
    # We iterate up to max_steps, but break internally when total_time is reached.
    pbar = tqdm(range(max_steps), desc="QITE Progress", unit="step")
    
    for step in pbar:
        # Termination Check
        if time_elapsed >= total_time:
            pbar.write("Total evolution time reached.")
            break

        step_accepted = False

        while not step_accepted:
            qc_trial = qc.copy()

            # Cap dt to hit total_time exactly
            if time_elapsed + current_dt > total_time:
                current_dt = total_time - time_elapsed
            
            # Perform QITE Step
            qc_trial, max_a = adaptive_qite_step(qc_trial, H, current_dt, domain_radius)
        
            # Measure Data
            psi_trial = Statevector.from_instruction(qc_trial)
            psi_trial = psi_trial / np.linalg.norm(psi_trial.data)
            new_energy = psi_trial.expectation_value(H).real
            energy_change = new_energy - current_energy

            # Convergence check feature
            if energy_change > -1e-5:
                print(f'Energy stalling or increasing after {step} iterations. Halving the time step.')
                current_dt *= 0.5
                if current_dt < min_dt:
                    pbar.write(f"Converged: Step size {current_dt:.2e} below threshold.")
                    return qc, energies, times, dts, max_as
            else:
                step_accepted = True
                qc = qc_trial
                current_energy = new_energy
                time_elapsed += current_dt
                energies.append(current_energy)
                times.append(time_elapsed)
                dts.append(current_dt)
                max_as.append(max_a)

                if filename is not None:
                    fout.write(f'{step+1:>10.0f} {current_dt:>10.4f} {current_energy:>10.6f} {energy_change:>10.6f} {max_a:>10.6f}\n')

                
                # Adaptive Logic for Next dt
                if max_a > 1e-9:
                    suggested_dt = target_rotation / max_a
                else:
                    suggested_dt = max_dt * 1.5 
                    
                # Smoothing & Limits
                suggested_dt = max(min(suggested_dt, current_dt * 2.0), current_dt * 0.5)
                current_dt = min(suggested_dt, max_dt)

                # Update Progress Bar Info
        pbar.set_description(f"T={time_elapsed:.2f}/{total_time} | E={current_energy:.4f} | dt={current_dt:.3f}")
        
    return qc, energies, times, dts, max_as

