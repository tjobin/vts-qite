
from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import HartreeFock
from utils import make_geometry
from qiskit_nature.second_q.mappers import JordanWignerMapper
import qiskit_nature.second_q.mappers as Mapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

# From IBM "Hamiltonians for Quantum Chemistry,
# https://quantum.cloud.ibm.com/learning/en/courses/quantum-chem-with-vqe/hamiltonian-construction
# Modified by Timothé Jobin


def build_hamiltonian_and_state(
        geometry : str,
        basis_set : str,
        n_elec : int,
        n_activeorbs : int,
        mapper : Mapper = JordanWignerMapper(),
        ) -> tuple[Statevector, SparsePauliOp]:
    """
    Returns the quantum circuit preparing the Hartree Fock state and the Hamiltonian of the system in the active space.
        Args: 
            - geometry : str, geometry string for the molecule. E.g. 'H 0 0 0; H 0 0 0.741;' in units Angstroms
            - basis_set : str, any from the pyscf basis set databank, e.g. 'sto-3g'
            - n_activeorbs : int, number of active orbitals in active space
            - n_elec : int, number of electrons used in the simulation, the rest is frozen     
            - mapper : qiskit_nature.second_q.mappers, fermion-to-qubit mapper, e.g. JordanWignerMapper() or ParityMapper()
        Returns:
            - state : qiskit.circuit.QuantumCircuit, the quantum circuit preparing the HF state
            - hamiltonian_full : qiskit.quantum_info.SparsePauliOp, the Hamiltonian of the system in the active space,
            including the nuclear repulsion energy and core electrons energies as a constant offset
    """
    
    driver = PySCFDriver(
        atom=geometry,
        basis=basis_set,
        charge=0, # assume neutral molecule
        spin=0    # assume singlet state
    )
    transformer = ActiveSpaceTransformer(
        num_electrons=n_elec,                # keep n_elec valence electrons
        num_spatial_orbitals=n_activeorbs      # keep n_activeorbs orbitals (e.g. n_activeorbs = 3 --> HOMO, LUMO, LUMO+1)
    )

    # Run the Driver to get the Electronic Problem
    problem = driver.run()
    reduced_problem = transformer.transform(problem)
    state = Statevector(HartreeFock(
        num_spatial_orbitals=reduced_problem.num_spatial_orbitals,
        num_particles=reduced_problem.num_particles,
        qubit_mapper=mapper
        ))

    # Generate the Qubit Hamiltonian
    # We use Jordan-Wigner mapping, which results in 2 * n_activeorbs qubits for STO-3G
    hamiltonian_op = mapper.map(reduced_problem.hamiltonian.second_q_op())

    # Add Nuclear Repulsion Energy (constant offset usually stored separately)
    # QITE minimizes the electronic part, but to match -1.137, we add this constant.
    nuclear_repulsion = reduced_problem.hamiltonian.nuclear_repulsion_energy
    core_energy = reduced_problem.hamiltonian.constants['ActiveSpaceTransformer']

    hamiltonian_full = hamiltonian_op + SparsePauliOp(["I" * hamiltonian_op.num_qubits], coeffs=[nuclear_repulsion+core_energy])

    return state, hamiltonian_full
