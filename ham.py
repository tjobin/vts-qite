import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import HartreeFock
from qiskit_algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem 
from qiskit_algorithms.time_evolvers.classical_methods import SciPyImaginaryEvolver
from pyscf_molecule import Molecule
from utils import make_geometry
from qiskit_nature.second_q.mappers import JordanWignerMapper

# From IBM "Hamiltonians for Quantum Chemistry,
# https://quantum.cloud.ibm.com/learning/en/courses/quantum-chem-with-vqe/hamiltonian-construction
# Modified by Timothé Jobin
    
def get_fermionic_ops(mol, aspace, get_ecore=True):
    mf = scf.RHF(mol)
    mf.scf()
    mx = mcscf.CASCI(mf, ncas=2, nelecas=(1, 1))
    E1 = mf.kernel()
    mx = mcscf.CASCI(mf, ncas=2, nelecas=(1, 1))
    mo = mx.sort_mo(aspace, base=0)
    E2 = mx.kernel(mo)[:2]
    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)
    if get_ecore==True:
        return h1e, h2e, ecore
    else:
        return h1e, h2e

def cholesky(V, eps):
    """The Cholesky function helps obtain a low-rank decomposition of
    the two-electron terms in the Hamiltonian.
    see https://arxiv.org/pdf/1711.02242.pdf section B2
    see https://arxiv.org/abs/1808.02625
    see https://arxiv.org/abs/2104.08957
    
    Args:
        - V : """
    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    while vmax > eps:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    L = L[:, :ng].reshape((no, no, ng))
    print(
        "accuracy of Cholesky decomposition ",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng

def identity(n):
    return SparsePauliOp.from_list([("I" * n, 1)])


def creators_destructors(n, mapping="jordan_wigner"):
    c_list = []
    if mapping == "jordan_wigner":
        for p in range(n):
            if p == 0:
                ell, r = "I" * (n - 1), ""
            elif p == n - 1:
                ell, r = "", "Z" * (n - 1)
            else:
                ell, r = "I" * (n - p - 1), "Z" * p
            cp = SparsePauliOp.from_list([(ell + "X" + r, 0.5), (ell + "Y" + r, 0.5j)])
            c_list.append(cp)
    else:
        raise ValueError("Unsupported mapping.")
    d_list = [cp.adjoint() for cp in c_list]
    return c_list, d_list

def build_hamiltonian(molecule, basis_set, spin=0, charge=0, mapping='jordan_wigner') -> SparsePauliOp:
    
    geometry = make_geometry(molecule)
    basis_set =  'ccpvdz'
    charge = 0
    spin = 0
    mol = Molecule(geometry, run_fci=False, basis=basis_set, unit='Bohr', charge=charge, spin=spin).mol
    active_space = range(mol.nelectron // 2 - 1, mol.nelectron // 2 + 1)

    h1e, h2e, ecore = get_fermionic_ops(mol, active_space)
    
    
    ncas, _ = h1e.shape

    C, D = creators_destructors(2 * ncas, mapping=mapping)
    Exc = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]]
        for r in range(p + 1, ncas):
            Excp.append(
                C[p] @ D[r]
                + C[ncas + p] @ D[ncas + r]
                + C[r] @ D[p]
                + C[ncas + r] @ D[ncas + p]
            )
        Exc.append(Excp)

    # low-rank decomposition of the Hamiltonian
    Lop, ng = cholesky(h2e, 1e-6)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)

    H = ecore * identity(2 * ncas)
    # one-body term
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p, r] * Exc[p][r - p]
    # two-body term
    for g in range(ng):
        Lg = 0 * identity(2 * ncas)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p, r, g] * Exc[p][r - p]
        H += 0.5 * Lg @ Lg

    return H.chop().simplify(), active_space